"""The loop's edge cases: python -m unittest test_agent_loop -v

smoke_test.py covers what the loop does on the real cohort. This covers
what it does when the brain misbehaves -- paths no seeded customer
reaches, and exactly the paths that matter once a real model is driving.

Every one of them must end in NO_ACTION. When we are not sure, we send
nothing.
"""

from __future__ import annotations

import unittest

from agent.loop import MAX_STEPS, Move, run_agent
from agent.schemas import Action, Cause, CustomerBrief, StepKind
from data.store import EventStore


class Brain:
    """Base test brain. Subclasses override next_move."""

    name = "test"

    def __init__(self) -> None:
        self.calls = 0
        self.errors: list[str] = []

    def reset(self) -> None:
        self.calls = 0

    def observe(self, step) -> None:
        pass

    def observe_error(self, message: str) -> None:
        self.errors.append(message)


BRIEF = CustomerBrief(customer_id="CUST-0001", probability=0.64, band="MEDIUM",
                      top_attributions=[], features={"days_of_supply_remaining": -20.0})


class AgentLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = EventStore.load().events_for("CUST-0001")

    def run_with(self, brain):
        return run_agent(brain, BRIEF, self.events)

    def test_escalation_ends_the_run_with_an_internal_note(self):
        """No seeded customer has three open complaints, so this branch is
        only ever exercised here."""

        class Escalator(Brain):
            def next_move(self, brief, steps):
                self.calls += 1
                if self.calls == 1:
                    return Move("Read the open tickets.", "read_support_tickets",
                                {"only_unresolved": True})
                return Move("Too messy for a template.", "escalate_to_human", {
                    "cause": "service_failure", "confidence": 0.9,
                    "rationale": "Several open complaints.",
                    "internal_note": "Call them before anything else goes out."})

        run = self.run_with(Escalator())
        self.assertEqual(run.diagnosis.action, Action.HUMAN_ESCALATION)
        self.assertEqual(run.stop_reason, "decided")
        self.assertEqual(run.steps[-1].kind, StepKind.DECIDE)
        self.assertIn("Call them", run.diagnosis.message_body)

    def test_tool_arguments_reach_the_tool(self):
        class Filtered(Brain):
            def next_move(self, brief, steps):
                self.calls += 1
                if self.calls == 1:
                    return Move("Open tickets only.", "read_support_tickets",
                                {"only_unresolved": True})
                return Move("Leave them.", "recommend_no_contact",
                            {"cause": "routine_lapse", "confidence": 0.5,
                             "rationale": "Nothing to fix."})

        run = self.run_with(Filtered())
        observation = run.investigation[0].observation
        self.assertEqual(observation["filter"], "unresolved only")
        self.assertTrue(all(not t["resolved"] for t in observation["tickets"]))

    def test_unknown_tool_is_fed_back_then_fails_quiet(self):
        class Confused(Brain):
            def next_move(self, brief, steps):
                return Move("Guessing.", "read_customer_mind", {})

        brain = Confused()
        run = self.run_with(brain)
        self.assertEqual(run.diagnosis.action, Action.NO_ACTION)
        self.assertEqual(run.diagnosis.cause, Cause.UNCLEAR)
        self.assertEqual(run.stop_reason, "too many bad calls")
        self.assertTrue(brain.errors and "read_customer_mind" in brain.errors[0])

    def test_bad_decision_arguments_are_explained_to_the_brain(self):
        class BadArgs(Brain):
            def next_move(self, brief, steps):
                return Move("Propose the wrong thing.", "propose_outreach", {
                    "cause": "routine_lapse", "confidence": 0.5,
                    "action": "no_action", "rationale": "x",
                    "subject": "y", "body": "z"})

        brain = BadArgs()
        run = self.run_with(brain)
        self.assertEqual(run.diagnosis.action, Action.NO_ACTION)
        self.assertIn("recommend_no_contact", brain.errors[0])

    def test_an_empty_message_body_is_rejected(self):
        class Empty(Brain):
            def next_move(self, brief, steps):
                return Move("Send nothing, badly.", "propose_outreach", {
                    "cause": "routine_lapse", "confidence": 0.5,
                    "action": "replenishment_reminder", "rationale": "x",
                    "subject": "y", "body": "   "})

        brain = Empty()
        run = self.run_with(brain)
        self.assertEqual(run.diagnosis.action, Action.NO_ACTION)
        self.assertIn("message body", brain.errors[0])

    def test_a_brain_that_never_decides_hits_the_step_limit(self):
        class Looper(Brain):
            def next_move(self, brief, steps):
                return Move("Again.", "check_product_supply", {})

        run = self.run_with(Looper())
        self.assertEqual(run.diagnosis.action, Action.NO_ACTION)
        self.assertEqual(run.stop_reason, "step limit")
        self.assertEqual(run.tool_calls, MAX_STEPS)

    def test_a_crashing_brain_does_not_take_the_cohort_down(self):
        class Broken(Brain):
            def next_move(self, brief, steps):
                raise RuntimeError("model unreachable")

        run = self.run_with(Broken())
        self.assertEqual(run.diagnosis.action, Action.NO_ACTION)
        self.assertEqual(run.stop_reason, "brain error")
        self.assertIn("model unreachable", run.diagnosis.reasoning_trace[-1])

    def test_confidence_is_clamped_rather_than_crashing(self):
        class Overconfident(Brain):
            def next_move(self, brief, steps):
                return Move("Certain.", "recommend_no_contact",
                            {"cause": "still_supplied", "confidence": 4.2,
                             "rationale": "They are stocked."})

        run = self.run_with(Overconfident())
        self.assertEqual(run.diagnosis.cause_confidence, 1.0)
        self.assertEqual(run.stop_reason, "decided")

    def test_the_agent_never_marks_its_own_work_as_sendable(self):
        class Sneaky(Brain):
            def next_move(self, brief, steps):
                return Move("Just send it.", "propose_outreach", {
                    "cause": "routine_lapse", "confidence": 0.5,
                    "action": "replenishment_reminder",
                    "rationale": "They ran out.", "subject": "Hi",
                    "body": "You are probably out.",
                    "requires_human_approval": False})

        run = self.run_with(Sneaky())
        self.assertTrue(run.diagnosis.requires_human_approval)


if __name__ == "__main__":
    unittest.main()
