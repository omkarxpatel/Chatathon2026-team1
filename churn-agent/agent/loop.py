"""The agent loop. This is the agentic part, and it is twenty lines long.

    brief in  ->  [ think -> call a tool -> read the result ] x N  ->  decide

The brain is asked what to do next, given only the brief and what it has
found so far. It answers with a tool call. If that tool reads something,
we run it, hand the result back, and ask again. If that tool is a
decision, the run ends and the decision becomes the recommendation.

Nothing in here knows whether the brain is Claude or the rule engine, and
that is the point: agent/claude_agent.py and agent/rules_agent.py sit
behind the same two-method protocol, take the same tools, and produce the
same AgentRun. The transcript the user reads on screen looks the same
either way, so the demo path is a fair picture of the real one.

Three safety properties, all enforced here rather than trusted to the brain:

  * the loop is bounded -- MAX_STEPS calls, then it stops
  * a bad tool call is an error handed back, not a crash
  * running out of steps produces NO_ACTION, never a guess
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import config as cfg
from agent import tools
from agent.schemas import (
    Action,
    AgentDiagnosis,
    AgentRun,
    AgentStep,
    AttributionInput,
    Cause,
    CustomerBrief,
    StepKind,
)
from data.store import CustomerEvents
from scoring.base import RiskResult

MAX_STEPS = 8


@dataclass
class Move:
    """What the brain wants to do next."""

    thought: str
    tool: str
    arguments: dict = field(default_factory=dict)


class Brain(Protocol):
    """Anything that can drive the loop.

    `next_move` is asked for one tool call at a time. `observe` hands the
    result back; brains that keep a conversation (the Claude one) append
    it, brains that do not (the rule one) can ignore it.
    """

    name: str

    def reset(self) -> None:
        """Start a fresh episode. One brain instance serves the whole
        cohort, so anything it remembers about the last customer has to
        go before the next one."""
        ...

    def next_move(self, brief: CustomerBrief, steps: list[AgentStep]) -> Move:
        ...

    def observe(self, step: AgentStep) -> None:
        ...

    def observe_error(self, message: str) -> None:
        ...


# --------------------------------------------------------------------------
# the brief
# --------------------------------------------------------------------------


def build_brief(risk: RiskResult, features: dict[str, float], top_n: int = 5) -> CustomerBrief:
    """What the agent starts with: the score and what drove it. No evidence.

    The agent has to go and get the evidence itself, through tools. That
    is not ceremony -- it is what makes the investigation on screen real,
    and it is what lets the agent stop early when the first tool call
    already answers the question.
    """
    return CustomerBrief(
        customer_id=risk.customer_id,
        probability=risk.probability,
        band=risk.band.value,
        top_attributions=[
            AttributionInput(feature=a.feature, label=a.label, value=a.value,
                             contribution=a.contribution, direction=a.direction)
            for a in risk.top(top_n)
        ],
        features=features,
    )


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def _trace(steps: list[AgentStep]) -> list[str]:
    """The investigation as a list of sentences, for the audit trail."""
    return [f"{s.thought} → {s.headline}" if s.headline else s.thought for s in steps]


def _gave_up(brief: CustomerBrief, brain: str, steps: list[AgentStep],
             why: str) -> AgentDiagnosis:
    """The only diagnosis the loop writes itself. Unsure means send nothing."""
    return AgentDiagnosis(
        customer_id=brief.customer_id,
        cause=Cause.UNCLEAR, cause_confidence=0.0,
        reasoning_trace=_trace(steps) + [why],
        action=Action.NO_ACTION,
        action_rationale=f"{why} Nothing is sent when the agent cannot reach a clear answer.",
        requires_human_approval=True, diagnoser=brain,
    )


def run_agent(brain: Brain, brief: CustomerBrief, ev: CustomerEvents) -> AgentRun:
    brain.reset()
    steps: list[AgentStep] = []
    errors = 0

    for _ in range(MAX_STEPS):
        try:
            move = brain.next_move(brief, steps)
        except Exception as exc:  # a brain that falls over must not take the cohort with it
            return AgentRun(customer_id=brief.customer_id, brain=brain.name, steps=steps,
                            diagnosis=_gave_up(brief, brain.name, steps,
                                               f"The agent failed mid-run ({exc})."),
                            stop_reason="brain error")

        tool = tools.TOOLS.get(move.tool)
        if tool is None:
            errors += 1
            brain.observe_error(f"No tool called {move.tool!r}. Available: "
                                f"{sorted(tools.TOOLS)}")
            if errors >= 3:
                break
            continue

        if tool.kind is StepKind.DECIDE:
            try:
                diagnosis = tools.build_decision(
                    move.tool, move.arguments, brief.customer_id, brain.name,
                    _trace(steps) + [move.thought])
            except tools.ToolError as exc:
                errors += 1
                brain.observe_error(str(exc))
                if errors >= 3:
                    break
                continue
            steps.append(AgentStep(index=len(steps) + 1, kind=StepKind.DECIDE,
                                   thought=move.thought, tool=move.tool,
                                   arguments=move.arguments,
                                   headline=diagnosis.action_rationale,
                                   observation={"action": diagnosis.action.value}))
            return AgentRun(customer_id=brief.customer_id, brain=brain.name, steps=steps,
                            diagnosis=diagnosis, stop_reason="decided")

        try:
            observation, headline = tools.investigate(move.tool, ev, move.arguments)
        except tools.ToolError as exc:
            errors += 1
            brain.observe_error(str(exc))
            if errors >= 3:
                break
            continue

        step = AgentStep(index=len(steps) + 1, kind=StepKind.INVESTIGATE,
                         thought=move.thought, tool=move.tool, arguments=move.arguments,
                         headline=headline, observation=observation)
        steps.append(step)
        brain.observe(step)

    why = ("The agent kept calling tools that do not exist."
           if errors >= 3 else
           f"The agent used all {MAX_STEPS} of its steps without reaching a decision.")
    return AgentRun(customer_id=brief.customer_id, brain=brain.name, steps=steps,
                    diagnosis=_gave_up(brief, brain.name, steps, why),
                    stop_reason="step limit" if errors < 3 else "too many bad calls")


# --------------------------------------------------------------------------
# which brain
# --------------------------------------------------------------------------


def get_brain() -> Brain:
    """One flag in config.py decides which brain runs.

    False -> agent/rules_agent.py, deterministic, no API key, no network.
    True  -> agent/claude_agent.py, a real Claude tool-use loop.
    """
    if cfg.USE_REAL_LLM:
        from agent.claude_agent import ClaudeBrain

        return ClaudeBrain()
    from agent.rules_agent import RuleBrain

    return RuleBrain()
