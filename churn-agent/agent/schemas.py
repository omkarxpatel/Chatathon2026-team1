"""Structured input and output for the agent layer.

Split into its own module so stub_llm.py and diagnose.py can be worked on
in parallel without fighting over the same file.

AgentDiagnosis is the contract. The stub produces it from rules, the real
model produces it from a tool-use loop, and everything downstream --
guardrails, pipeline, dashboard -- only ever sees this shape. That is what
makes swapping the LLM a one-line config change.
"""

from __future__ import annotations

from enum import Enum

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


class AttributionInput(BaseModel):
    """One model attribution, flattened for the agent."""

    feature: str
    label: str
    value: float
    contribution: float
    direction: str


class DiagnosisRequest(BaseModel):
    """Everything the agent is allowed to see.

    Note what is here and what is not. The agent gets the risk number --
    it does not compute it, and it cannot change it. It gets the model's
    attributions as evidence. It gets tool output, including the full
    text of support tickets. It does NOT get the churn label, the latent
    state, or any other customer's data.
    """

    customer_id: str
    probability: float
    band: str
    top_attributions: list[AttributionInput]
    features: dict[str, float]
    tool_output: dict


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
