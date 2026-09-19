# Cadence — word-for-word script

Companion to `PRESENTATION.md`, which keeps the running order, the cut list,
Q&A ownership and the pre-flight checklist. This file is only the words.

**How to use it.** Learn the shape, not the sentences. The **bolded** lines are
the ones to deliver close to verbatim — they carry the argument. Everything else
is scaffolding you can say in your own words. Bracketed italics are stage
directions, never spoken.

Timings assume ~145 words per minute. You will speak faster than that on the
day. That is what the 20-second buffer is for.

---

## Numbers on screen — read this first

The three tiles say **64 / 43 / 73**. They are three different denominators and
mixing them up in front of a judge is the one unforced error available to us.

| Tile | Means |
|---|---|
| Needs your review — **64** | acted on: a draft exists and a human must approve it |
| High-risk customers — **43** | HIGH band, regardless of what we decided to do |
| Outreach on hold — **73** | at-risk (not LOW) and deliberately not contacted |

**136** is the total left alone out of 200 — it is *not* on screen. M1 says it
while pointing at the 64, and pairs the two explicitly. Do not point at 73 and
say 136.

---

## M1 — The problem, and the claim · 0:00–0:40

*[Review queue, nothing selected. No introductions — open cold.]*

Customers buy on a rhythm — a subscription, a consumable, any reorder cycle.
Then they stop. Nobody complains, nobody unsubscribes; the next order just never
comes. You find out a quarter later.

The obvious build ranks everyone by risk and emails the top of the list. That's
what we deliberately did not build.

*[Point at the review queue count.]*

Two hundred customers. Sixty-four got a message. A hundred and thirty-six we
left alone on purpose.

> **Most of the customers our model flags should not be contacted. Not
> contacting someone is the default here, not the failure case.**

So — what earns a message? That starts with how the number gets made.

---

## M2 — The architecture, and why to believe it · 0:40–1:25

*[Still the cohort view. No clicking. Model stats visible.]*

Two layers, and the boundary between them is hard. None of this assumes what's
being sold — swap the catalogue and the split holds.

The deterministic layer does the arithmetic. Logistic regression, six features,
producing the probability and exact per-feature attributions — coefficient times
standardised value, not a SHAP approximation.

The agent layer reads what a regression can't — a support ticket is unstructured
text. It diagnoses *why*, and picks an action. Neither layer does the other's
job.

> **The LLM never computes the risk number. It receives the score and the
> attributions as input.**

On this seed: PR-AUC 0.825, precision-at-twenty eighty percent, against a
forty-seven percent base rate.

*[Say the metrics once. Do not dwell.]*

Let's watch it run on someone it decided to contact.

---

## M3 — CUST-0001, full trace · 1:25–2:30 · **driver**

*[Open demo case 1. Four tabs: Signals → Policy gate → Agent → Guardrails.
This is the only case where all four get walked.]*

Our data is a sports nutrition brand — two hundred synthetic customers.

CUST-0001. Sixty-four percent, medium risk.

*[Signals]* Their gap is 2.8 times their own median — and on serving count, they
ran out of product about twenty days ago.

*[Policy gate]* Four checks. Consent, cooldown, risk band, supply. All four
pass. Eligible.

*[Agent]* Service failure, eighty-eight percent confidence. Two damaged-shipment
tickets. The oldest is eighty-five days old and nobody ever answered it.

*[Read one line of the quoted ticket off the screen — it lands harder than
paraphrasing.]*

> **They didn't drift away. We broke something and then didn't answer. The right
> action is to fix it, not to discount it — a discount here buys silence.**

*[Guardrails]* All clear, awaiting human approval. And a discount requires a
price diagnosis — so the agent can't quietly staple "twenty percent off" onto a
service apology.

That one we acted on. Here's one the model also flagged, where we did nothing.

---

## M4 — CUST-0002, the gate · 2:30–3:25

*[Demo case 2. Go straight to the Policy gate tab. Do not open Agent — there is
nothing there, and that is the point.]*

CUST-0002. Thirty-six percent — lower than the customer we just emailed.

But the gap is 3.06 times their median — *bigger* than CUST-0001. On recency
alone, this person looks worse.

Here's what recency can't see. They bought a five-pound tub — seventy-six
servings, one a day. Three weeks of product left. Nothing is wrong yet.

So the gate stops them. Not the model — the gate. We enforce that as a hard rule
rather than ask a regression to estimate a fact we can compute.

And there's no agent output here. The gate runs first, so the agent never ran.
No tokens spent on someone we shouldn't contact.

> **A fifty-five day gap means nothing on a seventy-six day tub and everything on
> a fifteen day one. The model handles uncertainty; the gate handles certainty.**

Deterministic layer catching what the model missed. Now the other direction.

---

## M5 — CUST-0005, the agent, caveats, close · 3:25–4:40

*[Demo case 5. Policy gate tab briefly, then Agent.]*

CUST-0005. Eighty-five percent. The highest-risk customer we'll show you.

*[Policy gate — let it sit for a beat.]* Every check passes. Consent is fine. No
recent contact. High risk. Out of product. Every rule we have says email this
person, right now. A rules-only system does exactly that.

*[Agent]* Then the agent reads a support ticket from seventy-eight days ago.
Torn rotator cuff. Out of the gym for three months. CSAT five and five, zero
complaints. Diagnosis: lifestyle change. Action: no action.

A correct score and a correct decision, pointing in opposite directions.

> **The model says eighty-five percent and the model is right — they have
> churned. Emailing them protein anyway is the failure mode this whole system
> exists to avoid.**

*[Caveats. One breath, before a judge raises them. Do not sound defensive —
this is confidence, not apology.]*

Three things we'd want you to hold against this. It's synthetic data with a
generator we wrote, so 0.825 is ground-truth recovery, not a claim about real
customers. There's no uplift modelling — we rank who's at risk, not who's
saveable. And it's a single split on two hundred rows.

*[Close on restraint. Land it and stop — do not trail off into features.]*

There is no send path anywhere in this project. Every message is simulated, and
every one of them waits for a human.

---

## Delivery notes

**Handoffs.** Each segment ends on its own handoff line. Don't add "and now
I'll hand over to —". The line *is* the handoff; the next person just starts.

**The three pauses that matter.**
1. M1, after "on purpose" — let the asymmetry register before the thesis.
2. M4, after "the gate stops them. Not the model — the gate."
3. M5, on the Policy gate, before switching to Agent. This is the whole pitch.
   Give it a full beat and let the room get ahead of you.

**Numbers out loud.** Say "3.06 times" and "2.8 times", not "three point oh six
times their median value". Round in speech, precise on screen.

**If something breaks**, the named recovery person keeps talking while the
driver fixes it. Most of this argument survives without the screen — M4 and M5
in particular are stories, not demos.
