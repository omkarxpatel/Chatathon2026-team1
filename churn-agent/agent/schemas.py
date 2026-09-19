"""The contracts the agent layer speaks in.

Read this file first. It describes, in order, what the agent is given,
what it does, and what it hands back:

    CustomerBrief   what we put in front of the agent to start with --
                    the risk score (already computed, not negotiable)
                    plus what drove it. No evidence: the agent has to go
                    and get that itself.

    AgentStep       one turn of the loop. The agent said something, called
                    a tool, and got an answer back. The dashboard renders
                    these in order, and that list IS the explanation we
                    show the user.

    AgentDiagnosis  the conclusion: cause, action, and the drafted words.

    AgentRun        the whole episode -- brief in, steps, diagnosis out.

Split into its own module so the two brains (agent/rules_agent.py and
agent/claude_agent.py) can be worked on in parallel without fighting over
the same file, and so guardrails/pipeline/dashboard depend on the shapes
rather than on either implementation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Cause(str, Enum):
    """Why this customer is lapsing. Diagnosing this is the agent's real
    job -- the risk score says who, not why, and the why is what picks
    the action."""

    SERVICE_FAILURE = "service_failure"                  # we broke something
    PRICE_SENSITIVITY = "price_sensitivity"              # waiting for a deal
    LIFESTYLE_CHANGE = "lifestyle_change"                # injury, moved, stopped training
    STILL_SUPPLIED = "still_supplied"                    # not actually lapsed
    PRODUCT_DISSATISFACTION = "product_dissatisfaction"  # taste, mixability, results
    ROUTINE_LAPSE = "routine_lapse"                      # drifted, no specific cause
    UNCLEAR = "unclear"                                  # not enough signal


class Action(str, Enum):
    """What to do about it. Ordered roughly from least to most intrusive.

    NO_ACTION is a first-class outcome, not a failure. For a customer who
    has just told support they are injured, doing nothing is the correct
    and only decent answer.
    """

    NO_ACTION = "no_action"
    REPLENISHMENT_REMINDER = "replenishment_reminder"
    PRODUCT_GUIDANCE = "product_guidance"
    VALUE_EDUCATION = "value_education"        # cost per serving, not a % off
    SERVICE_RECOVERY = "service_recovery"
    LOYALTY_RECOGNITION = "loyalty_recognition"
    HUMAN_ESCALATION = "human_escalation"
    DISCOUNT_OFFER = "discount_offer"          # last resort; guardrails police this


# Actions that never produce a customer-facing message.
SILENT_ACTIONS = {Action.NO_ACTION}


# --------------------------------------------------------------------------
# what goes in
# --------------------------------------------------------------------------


class AttributionInput(BaseModel):
    """One model attribution, flattened for the agent."""

    feature: str
    label: str
    value: float
    contribution: float
    direction: str


class CustomerBrief(BaseModel):
    """The agent's starting position.

    Note what is here and what is not. The agent gets the risk number --
    it does not compute it and it cannot change it -- plus the model's
    attributions as a hint about where to look. It does NOT get the
    evidence. Ticket text, order history and supply arithmetic all arrive
    through tool calls the agent chooses to make, which is what makes the
    investigation on screen a real one.

    It never sees the churn label, the latent state, or another customer.
    """

    customer_id: str
    probability: float
    band: str
    top_attributions: list[AttributionInput]
    features: dict[str, float]


# --------------------------------------------------------------------------
# what happens in between
# --------------------------------------------------------------------------


class StepKind(str, Enum):
    INVESTIGATE = "investigate"   # pulled evidence
    DECIDE = "decide"             # committed to an answer and ended the run


class AgentStep(BaseModel):
    """One turn: the agent's stated intent, the tool it reached for, and
    what came back.

    `headline` is the one-line, plain-language version of the finding.
    It is written by the tool, not by the agent, so it is true even when
    the agent misreads its own evidence -- which is exactly when a
    reviewer needs it most.
    """

    index: int
    kind: StepKind
    thought: str                     # why the agent made this call
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    headline: str                    # what came back, in one line
    observation: dict[str, Any] = Field(default_factory=dict)


class AgentDiagnosis(BaseModel):
    customer_id: str

    cause: Cause
    cause_confidence: float = Field(ge=0.0, le=1.0)
    # Ordered steps, shown verbatim in the dashboard. This is the audit
    # trail: if the action is wrong, the trace shows where it went wrong.
    reasoning_trace: list[str]

    action: Action
    action_rationale: str

    message_subject: str | None = None
    message_body: str | None = None

    # There is no send path in this project, and this flag is checked by
    # the guardrails. Nothing reaches a customer without a human.
    requires_human_approval: bool = True

    diagnoser: str = "unknown"

    @property
    def has_message(self) -> bool:
        return bool(self.message_body)


class AgentRun(BaseModel):
    """One complete episode of the loop, start to finish.

    The dashboard reads this directly: `steps` is the investigation the
    user watches, `diagnosis` is the recommendation they act on.
    """

    customer_id: str
    brain: str                       # which brain ran: "rules" or a Claude model id
    steps: list[AgentStep]
    diagnosis: AgentDiagnosis
    stop_reason: str

    @property
    def investigation(self) -> list[AgentStep]:
        return [s for s in self.steps if s.kind == StepKind.INVESTIGATE]

    @property
    def tool_calls(self) -> int:
        return len(self.steps)

    def evidence(self) -> dict[str, Any]:
        """Every observation the agent gathered, keyed by tool name.

        Last call wins, which is what you want when a tool was called
        twice with different arguments.
        """
        return {s.tool: s.observation for s in self.investigation}
