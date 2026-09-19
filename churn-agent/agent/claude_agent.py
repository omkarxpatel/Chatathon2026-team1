"""The real agent: Claude, with tools, in a loop.

One flag in config.py swaps this in for the rule brain. Nothing else
changes, because the loop, the tools, the guardrails and the dashboard
all sit behind the same contracts.

What Claude is and is not allowed to do here is worth being precise about:

  it is NOT given the risk number to compute      -- scoring/ did that
  it is NOT given the evidence up front           -- it has to fetch it
  it is NOT given a send path                     -- there isn't one
  it IS given the unstructured signals            -- ticket text above all
  it IS given the decision                        -- cause, action, copy

Each turn it writes one sentence about what it is doing and calls one
tool. agent/loop.py runs the tool, hands the result back, and asks again.
The run ends when it calls one of the three decision tools, or when the
step limit stops it -- and hitting the limit produces NO_ACTION, not a
guess.
"""

from __future__ import annotations

import json

import config as cfg
from agent import tools
from agent.loop import Move
from agent.schemas import AgentStep, CustomerBrief

SYSTEM_PROMPT = f"""\
You work retention for a direct-to-consumer sports nutrition brand (whey, \
creatine, pre-workout, electrolytes, multivitamins).

A statistical model has already scored this customer's churn risk. That \
number is given to you and is not yours to revise or second-guess. Your job \
is the part the model cannot do: work out WHY, using evidence you go and \
fetch, then decide what -- if anything -- we should do about it.

Work one step at a time. Each turn: write ONE short sentence saying what you \
are checking and why, then call exactly one tool. Keep going until you have \
enough to decide, then call one of the three decision tools. Do not call a \
tool you have already called unless the arguments are genuinely different.

How to think about it:

- Products have a computable depletion date. A 30-serving tub at one scoop a \
  day runs out in 30 days. A long gap after a 76-day pack is arithmetic, not \
  disengagement. Call check_product_supply first; if they are still stocked, \
  stop there and recommend no contact.
- If we caused the problem -- damaged shipment, ignored ticket, billing error \
  -- fix that. Do not attach an offer. A discount on top of an unanswered \
  complaint reads as buying silence.
- recommend_no_contact is a correct and common answer. If someone has told \
  support they are injured and out of the gym, they have not lapsed, they \
  have stopped, and emailing them protein is tasteless. Say so plainly.
- Reach for discount_offer almost never. If price is genuinely the blocker, \
  prefer value_education: cost per serving, larger pack economics, \
  subscribe-and-save. A percentage off works once and teaches them to wait \
  for the next one.
- Email opens are inflated by Apple Mail Privacy Protection. Judge \
  engagement on clicks.

House style for any message you draft:
- Say why you are writing in the first line.
- No urgency, no countdowns, no "don't miss out", no exclamation marks.
- Never promise a discount unless the action is discount_offer.
- Always give them a way to hear from us less often.
- Under {cfg.MAX_MESSAGE_WORDS} words.

Your sentence-per-step and your rationale are shown to a human reviewer, and \
they are how that person decides whether to trust you. Quote the evidence you \
relied on. Every message is reviewed by a person before it goes anywhere. \
Nothing you write is sent automatically."""


def _render_brief(brief: CustomerBrief) -> str:
    attributions = "\n".join(
        f"  - {a.label} = {a.value:g}  ({a.direction}, weight {a.contribution:+.2f})"
        for a in brief.top_attributions)
    return (
        f"Customer: {brief.customer_id}\n"
        f"Model churn risk: {brief.probability:.0%} ({brief.band} band)\n\n"
        f"What drove that score (from the model, not from you):\n{attributions}\n\n"
        f"Start investigating.")


class ClaudeBrain:
    """Same interface as the rule brain. See agent/loop.py for the protocol."""

    def __init__(self, model: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "config.USE_REAL_LLM is True but the `anthropic` package is not "
                "installed. Run `pip install anthropic`, or set USE_REAL_LLM = "
                "False to use the rule-based agent."
            ) from exc
        # Resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
        # `ant auth login` profile, in that order.
        self.client = anthropic.Anthropic()
        self.model = model or cfg.LLM_MODEL
        self.name = self.model
        self.messages: list[dict] = []
        self._tool_use_id: str | None = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """One brain serves the whole cohort; each customer gets a clean
        conversation. Carrying one customer's tickets into the next
        customer's reasoning would be a privacy bug, not a feature."""
        self.messages = []
        self._tool_use_id = None

    def next_move(self, brief: CustomerBrief, steps: list[AgentStep]) -> Move:
        if not self.messages:
            self.messages = [{"role": "user", "content": _render_brief(brief)}]

        for attempt in range(3):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=cfg.LLM_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=tools.TOOL_SCHEMAS,
                messages=self.messages,
            )
            self.messages.append({"role": "assistant", "content": response.content})

            thought = " ".join(b.text.strip() for b in response.content
                               if b.type == "text" and b.text.strip())
            call = next((b for b in response.content if b.type == "tool_use"), None)
            if call is not None:
                self._tool_use_id = call.id
                label = tools.TOOL_LABELS.get(call.name, call.name)
                return Move(thought=thought or f"{label}.", tool=call.name,
                            arguments=dict(call.input or {}))

            # It answered in prose. Ask again; the loop only moves on tool calls.
            self._tool_use_id = None
            self.messages.append({"role": "user", "content":
                                  "Call a tool. If you have enough to decide, call one "
                                  "of propose_outreach, recommend_no_contact or "
                                  "escalate_to_human."})

        raise RuntimeError("Claude did not call a tool after three attempts.")

    # ------------------------------------------------------------------
    def observe(self, step: AgentStep) -> None:
        self._result(json.dumps(step.observation, indent=2, default=str))

    def observe_error(self, message: str) -> None:
        self._result(f"Tool error: {message}", is_error=True)

    def _result(self, content: str, is_error: bool = False) -> None:
        if self._tool_use_id is None:
            self.messages.append({"role": "user", "content": content})
            return
        self.messages.append({"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": self._tool_use_id,
            "content": content, "is_error": is_error,
        }]})
        self._tool_use_id = None
