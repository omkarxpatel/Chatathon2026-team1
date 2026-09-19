# 5 minutes, 5 speakers — running order

Chatathon 2026, Klaviyo track. Read your own card, then read the two either
side of yours so the handoffs are clean.

**The spine of the pitch:** three customers, each one justifying a different
layer of the system.

| | Customer | What it proves |
|---|---|---|
| Act 1 | CUST-0001 | the machine works end to end |
| Act 2 | CUST-0002 | **arithmetic beats the model** — the gate stops someone the model flagged |
| Act 3 | CUST-0005 | **text beats the rules** — the gate passes them, only the agent can stop it |

Acts 2 and 3 are mirror images. That is the argument. Do not let the demo
become "here are five customers."

---

## Timing

| Time | Who | Segment |
|---|---|---|
| 0:00 – 0:40 | **M1** | The problem, and the claim |
| 0:40 – 1:25 | **M2** | The architecture, and why to believe the number |
| 1:25 – 2:30 | **M3** | CUST-0001 — full trace |
| 2:30 – 3:25 | **M4** | CUST-0002 — the gate |
| 3:25 – 4:40 | **M5** | CUST-0005 — the agent, caveats, close |
| 4:40 – 5:00 | — | buffer |

**No introductions.** No "hi, we're team four, today we'll be showing you."
That costs 15 seconds and earns nothing. M1 opens cold on the problem. Names
go on the screen if they go anywhere.

**One driver, fixed at the keyboard, for all five minutes.** Suggest M3 — they
have the most screen work. Nobody else touches the laptop. Handing a laptop
between five people inside five minutes is where demos die.

---

## M1 — The problem, and the claim · 40s

**Owns:** why anyone should care, and the counterintuitive thesis.

**On screen:** cohort view, nothing selected. The headline tile is already up.

**Beats**
- Any repeat-purchase business. People lapse quietly; you find out when the
  reorder never comes. Name the vertical in M3, not here.
- The obvious build is "rank everyone by risk, email the top of the list."
- That is the thing we deliberately did not build.
- Point at the headline: **136 of 200 deliberately left alone. 64 got a message.**

**Must land, close to verbatim**
> Most of the customers our model flags should not be contacted. Not
> contacting someone is the default here, not the failure case.

**Handoff:** "…so the interesting question is what earns a message — and that
starts with how the number gets made."

---

## M2 — The architecture, and why to believe it · 45s

**Owns:** the one architectural rule, and the model's credibility.

**On screen:** still the cohort view — the funnel and the sidebar model stats
are both visible. No clicking.

**Beats**
- Two layers, hard boundary. Deterministic layer scores. Agent layer diagnoses.
- Structured behaviour goes to the model, which is good at it. Unstructured
  text — a support ticket — goes to the LLM, which is the only layer that can
  read it. Neither is asked to do the other's job.
- Logistic regression, six features, exact attributions — contribution is
  coefficient × standardised value, no SHAP approximation.
- Gesture at the sidebar: PR-AUC 0.825, precision@20 80%, against a 47% base
  rate. Say the numbers once, do not dwell.

**Must land, close to verbatim**
> The LLM never computes the risk number. It receives the score and the
> attributions as input.

**Handoff:** "So let's watch it run on someone it decided to contact."

---

## M3 — CUST-0001, full trace · 65s · **also the driver**

**Owns:** proof the whole pipeline works. This is the only case where all four
tabs get walked. Later acts are single-tab visits.

**On screen:** select demo case 1. Then Signals → Policy gate → Agent →
Guardrails.

**Beats**
- **Signals:** 64%, MEDIUM. Gap is 2.81× their own median, and they ran out of
  product about 20 days ago.
- **Policy gate:** all four checks pass — consent, cooldown, risk band, supply.
  Eligible.
- **Agent:** diagnosed `service_failure` at 88% confidence. Two unresolved
  damaged-shipment tickets, oldest one 85 days old, never answered. Read one
  line of the quoted ticket off the trace — it lands harder than paraphrasing.
- **Guardrails:** all clear, awaiting human approval. Note in one breath that a
  discount requires a price diagnosis, so the agent cannot quietly staple
  "20% off" onto a service apology.

**Must land, close to verbatim**
> They did not drift away. We broke something and then did not answer. The
> right action is to fix it, not to discount it — a discount here buys silence.

**Handoff:** "That one we acted on. Here's one the model also flagged — and we
did nothing."

---

## M4 — CUST-0002, the gate · 55s

**Owns:** arithmetic beating the model. The false positive.

**On screen:** select demo case 2, go straight to the **Policy gate** tab.
There is no Agent tab content here — the gate stopped it, so the agent never
ran. Say that out loud; it is the point.

**Beats**
- 36% risk — but the gap is **3.06× their median, bigger than the customer we
  just emailed**. On recency alone this person looks worse than CUST-0001.
- They bought a 5 lb tub. Servings ÷ servings-per-day says they still have
  about 21 days of product. Nothing is wrong yet.
- So the gate stops them. Not the model — the gate. This is not a statistical
  correlate, it is arithmetic, and we enforce it as a hard rule rather than
  asking a regression to estimate a fact we can compute.
- The agent was never invoked. No tokens spent reasoning about someone we
  should not contact.
- One sentence for depth, no clicking: two more cases in the build — one where
  we're inside the 14-day cooldown, one where the answer is cost-per-serving
  maths rather than a discount.

**Must land, close to verbatim**
> A 55-day gap means nothing on a 76-day tub and everything on a 15-day one.
> The model handles uncertainty; the gate handles certainty.

**Handoff:** "That's the deterministic layer catching what the model missed.
Now the other direction."

---

## M5 — CUST-0005, the agent, caveats, close · 75s

**Owns:** the climax, the honesty, and the last word. Most time, because this
is the segment that gets remembered.

**On screen:** select demo case 5. **Policy gate** tab briefly, then **Agent**.

**Beats**
- 85%, HIGH. The highest-risk case we'll show.
- **Show the gate passing.** All four checks green — consent fine, no recent
  contact, high risk, out of product. Every rule we have says contact this
  person. A rules-only system emails them right here.
- Switch to Agent. It read a support ticket from 78 days ago: torn rotator
  cuff, out of the gym three months. CSAT history 5 and 5. No complaints.
  Diagnosed `lifestyle_change`. Action: `no_action`.
- The model is not wrong — they have churned. The decision is still to do
  nothing. A correct score and a correct decision pointing in opposite
  directions.
- **Caveats, one breath, before a judge raises them:** synthetic data with a
  known generator, so 0.825 is ground-truth recovery and not a claim about
  real customers. No uplift modelling — we rank who's at risk, not who's
  saveable. Single split on 200 rows.
- Close on restraint, not on features. There is no send path anywhere in the
  project — every message is simulated and every one requires human approval.

**Must land, close to verbatim**
> The model says 85% and the model is right — they have churned. Emailing them
> protein anyway is the failure mode this whole system exists to avoid.

---

## If you're running long, cut in this order

1. The Guardrails tab in M3 (−10s) — fold it into one spoken clause.
2. M4's mention of the two extra cases (−8s).
3. M2's metric read-out, keep only PR-AUC (−7s).
4. M3's ticket quote, paraphrase instead (−8s).

**Never cut:** CUST-0005, the days-of-supply explanation, "the LLM never
computes the risk number," or the caveats. Those four are the pitch.

---

## Q&A ownership

Decide this now so nobody talks over anybody.

| Likely question | Answers |
|---|---|
| "Isn't this just a wrapper around an LLM?" | **M2** |
| "How do you know the model isn't overfit?" | **M2** |
| "Why not just rank by recency?" | **M4** — that is literally CUST-0002 |
| "What if the LLM hallucinates or goes off-tone?" | **M5** — guardrails run on output |
| "Does this actually send email?" | **M3** — no send path exists, by design |
| "Would this work on real data?" | **M1** — owns the honest answer |

---

## Before you present

- [ ] **Re-check the keystrokes after the GUI work lands.** The demo-case list
      and arrow-key stepping are being changed by the other dev this week.
      Whoever drives should re-walk the run once on the final build.
- [ ] Run `python smoke_test.py` on the demo machine. It asserts all five
      cases behave as claimed — if it passes, the demo cannot surprise you.
- [ ] Start the app before you're in the room. First load scores the cohort
      and runs the agent; it is not instant.
- [ ] Rehearse twice with a stopwatch, full run, no stopping. Five speakers
      overrun on the first attempt essentially every time.
- [ ] Agree who recovers if something breaks: one named person keeps talking
      while the driver fixes it.
