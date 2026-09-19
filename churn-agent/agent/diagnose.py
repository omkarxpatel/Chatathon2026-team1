"""The agent layer: risk score + evidence -> cause, action, and copy.

This is the only place in the project allowed to call an LLM, and note
what it is NOT allowed to do: it never computes the risk number. The
probability arrives already calculated by scoring/, and the agent's job
starts after that. It reads the unstructured signals a logistic
regression cannot -- ticket text, the shape of an order history -- works
out WHY, picks an action, and drafts the words.

Two implementations, one interface:

    StubDiagnoser      rules, deterministic, no API key, no network
    AnthropicDiagnoser Claude, via the official SDK

Flip config.USE_REAL_LLM to swap. Both return AgentDiagnosis, so the
policy gate, the guardrails, and the dashboard cannot tell them apart.
"""

from __future__ import annotations

from typing import Protocol

import config as cfg
from agent import tools
from agent.schemas import (
    Action,
    AgentDiagnosis,
    AttributionInput,
    Cause,
    DiagnosisRequest,
)
from data.store import CustomerEvents
from scoring.base import RiskResult


class Diagnoser(Protocol):
    name: str

    def diagnose(self, request: DiagnosisRequest) -> AgentDiagnosis:
        ...


# --------------------------------------------------------------------------
# building the request
# --------------------------------------------------------------------------


def build_request(
    risk: RiskResult, features: dict[str, float], ev: CustomerEvents, top_n: int = 5
) -> DiagnosisRequest:
    """Assemble everything the agent is allowed to see.

    Tool output is fetched up front rather than through a live tool-call
    loop. Two reasons: the stub and the real model then reason over
    byte-identical evidence, which makes the stub a fair stand-in; and
    there are only five tools, so there is nothing to save by calling
    them lazily. TOOL_SCHEMAS in agent/tools.py is already in Anthropic
    tool-use format if you want to make it a real loop later.
    """
    return DiagnosisRequest(
        customer_id=risk.customer_id,
        probability=risk.probability,
        band=risk.band.value,
        top_attributions=[
            AttributionInput(
                feature=a.feature, label=a.label, value=a.value,
                contribution=a.contribution, direction=a.direction,
            )
            for a in risk.top(top_n)
        ],
        features=features,
        tool_output=tools.run_all(ev),
    )


# --------------------------------------------------------------------------
# the real model
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You work retention for a direct-to-consumer sports nutrition brand (whey, \
creatine, pre-workout, electrolytes, multivitamins).

A statistical model has already scored this customer's churn risk. That \
number is given to you and is not yours to revise or second-guess. Your job \
is the part the model cannot do:

1. Work out WHY this customer is lapsing, using the unstructured evidence -- \
   support ticket text above all, plus the shape of their order history.
2. Choose an action that fits the cause, not the score.
3. Draft the message, if a message is warranted.

How to think about it:

- Products have a computable depletion date. A 30-serving tub at one scoop a \
  day runs out in 30 days. A long gap after a 76-day pack is arithmetic, not \
  disengagement. Check get_product_usage before you conclude anything.
- If we caused the problem -- damaged shipment, ignored ticket, billing error \
  -- fix that. Do not attach an offer. A discount on top of an unanswered \
  complaint reads as buying silence.
- NO_ACTION is a correct and common answer. If someone has told support they \
  are injured and out of the gym, they have not lapsed, they have stopped, \
  and emailing them protein is tasteless. Leaving them alone is the right \
  call and you should say so plainly.
- Reach for DISCOUNT_OFFER almost never. If price is genuinely the blocker, \
  prefer VALUE_EDUCATION: cost per serving, larger pack economics, \
  subscribe-and-save. A percentage off works once and teaches them to wait \
  for the next one.
- Email opens are inflated by Apple Mail Privacy Protection. Judge engagement \
  on clicks.

House style for any message you draft:
- Say why you are writing in the first line.
- No urgency, no countdowns, no "don't miss out", no exclamation marks.
- Never promise a discount unless the action is DISCOUNT_OFFER.
- Always give them a way to hear from us less often.
- Under 130 words.

Fill reasoning_trace with the ordered steps you actually took, each a short \
sentence, quoting the evidence you relied on. It is shown to a human \
reviewer, and it is how they decide whether to trust the action.

Every message is reviewed by a person before it goes anywhere. Nothing you \
write is sent automatically."""


def _render_request(request: DiagnosisRequest) -> str:
    import json

    attributions = "\n".join(
        f"  - {a.label} = {a.value:g}  ({a.direction}, weight {a.contribution:+.2f})"
        for a in request.top_attributions
    )
    return (
        f"Customer: {request.customer_id}\n"
        f"Model churn risk: {request.probability:.0%} ({request.band} band)\n\n"
        f"What drove that score (from the model, not from you):\n{attributions}\n\n"
        f"Evidence from the tools:\n"
        f"{json.dumps(request.tool_output, indent=2, default=str)}\n\n"
        f"Diagnose the cause, choose an action, and draft the message if one "
        f"is warranted."
    )


class AnthropicDiagnoser:
    """Claude via the official SDK, with a schema-validated response.

    Structured output through client.messages.parse() rather than a tool-
    call loop: we want one guaranteed-shaped answer, not a conversation.
    """

    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "config.USE_REAL_LLM is True but the `anthropic` package is not "
                "installed. Run `pip install anthropic`, or set USE_REAL_LLM = "
                "False to use the stub."
            ) from exc
        # Resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
        # `ant auth login` profile, in that order.
        self.client = anthropic.Anthropic()
        self.model = model or cfg.LLM_MODEL

    def diagnose(self, request: DiagnosisRequest) -> AgentDiagnosis:
        from pydantic import BaseModel, Field

        class LLMDiagnosis(BaseModel):
            """What the model fills in. customer_id and the approval flag
            are set by us afterwards -- they are not the model's to choose."""

            cause: Cause
            cause_confidence: float = Field(ge=0.0, le=1.0)
            reasoning_trace: list[str]
            action: Action
            action_rationale: str
            message_subject: str | None = None
            message_body: str | None = None

        response = self.client.messages.parse(
            model=self.model,
            max_tokens=cfg.LLM_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _render_request(request)}],
            output_format=LLMDiagnosis,
        )
        parsed = response.parsed_output

        return AgentDiagnosis(
            customer_id=request.customer_id,
            cause=parsed.cause,
            cause_confidence=parsed.cause_confidence,
            reasoning_trace=parsed.reasoning_trace,
            action=parsed.action,
            action_rationale=parsed.action_rationale,
            message_subject=parsed.message_subject,
            message_body=parsed.message_body,
            # Not negotiable, and not the model's decision.
            requires_human_approval=True,
            diagnoser=f"{self.name}:{self.model}",
        )


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def get_diagnoser() -> Diagnoser:
    """One flag in config.py decides which brain runs."""
    if cfg.USE_REAL_LLM:
        return AnthropicDiagnoser()
    from agent.stub_llm import StubDiagnoser

    return StubDiagnoser()
