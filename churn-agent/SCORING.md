# How scoring works

Plain-English walkthrough of how a customer goes from raw events to a
decision. Read this before touching `features/`, `scoring/`, or `agent/`.

> Numbers below are from the committed seed (`config.SEED = 20260919`).
> Regenerate and they shift a little. `python run_pipeline.py` prints live values.

---

## The one-paragraph version

We take everything a customer did — orders, site visits, emails, support
tickets — and squeeze it into **six numbers**. A logistic regression turns those
six numbers into **one probability**. That probability does *not* decide
anything on its own: it just earns the customer the right to be *considered*.
Four deterministic rules then remove most of them, and only what survives goes
to the agent, which reads the actual support tickets and decides what (if
anything) to do.

Of 200 customers, **136 are deliberately left alone**. That is the system
working, not failing.

---

## Step 1 — Events become numbers

`features/extract.py`

Every customer has a pile of raw events. We compute **29 features** from them,
but only **6 are fed to the model**. The rest are computed and displayed in the
dashboard so a human can see them, but withheld from the scorer.

The six the model actually sees:

| Feature | What it means in English |
|---|---|
| `reorder_gap_ratio` | How overdue they are **relative to their own normal**. 1.0 = right on schedule. 3.0 = three times their usual gap. |
| `click_rate_90d` | Share of delivered emails they clicked, last 90 days. |
| `days_since_last_session` | Days since they last visited the site. |
| `sessions_since_last_order` | How many times they've browsed since last buying. |
| `promo_order_share` | Share of their orders that used a promo code. |
| `order_count_lifetime` | How many orders they've placed in the window. |

**Two important absences.**

- **Days of supply left** is *not* a model feature, even though it is the most
  important number in the project. It's arithmetic — servings ÷ servings-per-day
  — not something to estimate. It's enforced as a hard rule in the gate instead.
- **Support tickets** are not model features either. "Two open tickets about a
  damaged shipment, photos sent, no reply in 85 days" is not a number. The agent
  reads them directly.

### Two gotchas that will bite you

**Undefined is not zero.** If someone has added nothing to a cart in 90 days,
their cart-abandon rate isn't `0.0` — it's *undefined*. We write `NaN` and fill
with the cohort median. Writing `0.0` made dead customers look like perfect
converters and **flipped the sign of the coefficient**. This is fixed, but don't
reintroduce it.

**Features never see the future.** `data/store.py` truncates every customer's
events at the cutoff date before the extractor gets them. The extractor does no
file I/O at all, so it physically cannot reach past that.

---

## Step 2 — Numbers become a score

`scoring/logistic.py`

It's a plain logistic regression. Written out in full:

```
L = 0.4474
  + 0.4324 × reorder_gap_ratio
  − 3.5187 × click_rate_90d
  + 0.0277 × days_since_last_session
  − 0.0410 × sessions_since_last_order
  + 0.8872 × promo_order_share
  − 0.2584 × order_count_lifetime

risk = 1 / (1 + e^−L)
```

Positive weight = pushes toward churn. Read the signs aloud — they should all
sound like common sense, and they do. If you ever change the feature set and a
sign stops making sense, **stop**, because the dashboard's attribution panel
will start telling the audience something false.

### Why only six features?

We have ~94 churn events. With 24 predictors that's under 4 events per
variable, against an accepted floor of 10. Below that floor, correlated features
trade weight and their coefficients flip sign — at one point the model was
claiming *more product on hand means more churn risk*. Six predictors is 15.7
events per variable and every sign behaves.

### Attributions are exact

For each customer we report how much each feature contributed. Because the
model is linear, contribution = `coefficient × standardised value`, and they sum
back to the log-odds exactly. `smoke_test.py` checks this reconstructs to within
1.7e-14. **No SHAP needed** — it isn't an approximation.

---

## Step 3 — Score becomes a band

Same cut points Klaviyo publishes, on purpose, so we're speaking their language:

| Band | Risk |
|---|---|
| LOW | under 33% |
| MEDIUM | 33% – 66% |
| HIGH | over 66% |

---

## Step 4 — How a customer actually gets flagged

This is the part people get wrong. **A high score does not mean we contact
someone.** Four deterministic rules run *before* the agent, in this order, and
the first failure decides. All of it lives in `agent/policy.py`.

### Rule 1 — Consent

*Did they unsubscribe or file a spam complaint?* If yes, permanently suppressed.
No score overrides this.

> **CUST-0161** — risk **99%**, hasn't visited in 132 days, 105 days out of
> product. Textbook churner. Unsubscribed on 2026-04-09, so we never contact
> them. Nothing about the score matters here.

### Rule 2 — Cooldown

*Have we sent them a targeted message in the last 14 days?*

> **CUST-0181** — risk **95%**, gap 5.1× their normal. Genuinely at risk and
> genuinely actionable. But a win-back went out 12 days ago, so it's suppressed
> and **the agent never runs** — we don't spend tokens reasoning about someone
> we're not allowed to contact.

Only **targeted flow** messages count (`winback_flow`, `replenishment_reminder`,
`vip_checkin`, `service_followup`). Broadcast newsletters don't. The brand emails
the whole list every ~6 days, so counting broadcasts would suppress 148 of 200
and make the rule meaningless theatre.

### Rule 3 — Risk band

*Is the score at least MEDIUM?* LOW-band customers are never contacted — there's
no problem to solve.

> **CUST-0154** — risk **33%**, clicking 24% of emails, visited 17 days ago.
> Fine. Leave them alone.

### Rule 4 — Supply

*Do they still have more than 7 days of product on hand?*

This is the product-physics rule and it's the one that makes this project
different. A 30-serving tub at one scoop a day runs out in 30 days. A 5 lb tub
lasts 76 days. A long gap on a big tub is **arithmetic, not disengagement**.

> **CUST-0007** — risk **89%**, hasn't visited in 92 days. Looks terrible. But
> their last order left them **+28 days of product**. They're not lapsed, they're
> just not due yet. Suppressed.

> **CUST-0002** (demo case) — gap is **3.06×** their median, *worse* than our
> flagship churner. They bought a 5 lb tub. ~21 days of protein left. They
> reorder on schedule inside the forward window. Naive recency ranks them #99 of
> 200; this rule stops them dead.

### What survives

```
200  Scored
183  Have not opted out          17 removed by consent
134  Not contacted recently      49 removed by cooldown
 96  At meaningful risk          38 removed by risk band
 67  Actually out of product     29 removed by supply
 64  Agent chose to act           3 left alone by the agent
```

Notice that **the cooldown and supply rules remove more people than the risk
model does.**

---

## Step 5 — The agent decides what to do

`agent/stub_llm.py` (rules) or `agent/diagnose.py` (real Claude). Both return the
same `AgentDiagnosis` object, so nothing downstream can tell them apart.

The agent gets the score and the attributions **as input**. It never recomputes
them. Its job is the part a regression can't do: read the support tickets, work
out *why*, pick an action, write the words.

It checks in this order:

1. **Still supplied?** → no action.
2. **Did we break something?** Unresolved negative tickets → service recovery.
   No discount attached; a discount on top of an unanswered complaint reads as
   buying silence.
3. **Have they told us they stopped?** Injury, moving, pregnancy → **no action**.
4. **Are they blocked on price?** High promo share + browsing without buying →
   value education (cost per serving, subscribe-and-save) — **not** a percentage off.
5. **Did the product disappoint?** → product guidance.
6. **None of the above** → a low-key replenishment reminder.

### Real examples

> **CUST-0137** — risk 70%. Gap is only 0.43× (not even overdue), but there are
> unresolved negative tickets. → `service_failure` → **service recovery**.

> **CUST-0072** — risk 95%. Gap 0.74× — again, *not overdue*. But **100% of their
> orders used a promo** and they haven't visited in 72 days. → `price_sensitivity`
> → **value education**.

> **CUST-0037** — risk 99%. Gap **8×**, 180 days out of product, one lifetime
> order, no complaints, no life event. → `routine_lapse` → **replenishment reminder**.

> **CUST-0005** — risk **85%**, and the model is right; they have churned. But
> their last ticket says *"I've torn my rotator cuff and I'm out of the gym for at
> least three months"*, CSAT 5/5, zero complaints. → `lifestyle_change` → **no
> action**. Selling protein to someone who just told support they're injured is
> the failure mode this whole system exists to avoid.

Note that CUST-0137 and CUST-0072 were both **barely overdue**. The score picked
them up on engagement decay, and the *ticket text* is what explained them. That's
the two-layer split doing its job.

---

## Step 6 — Guardrails check the agent's homework

`agent/guardrails.py`. These run on the agent's **output**. A failure downgrades
the action, usually to "do nothing" — when unsure, send nothing.

The important ones:

- **A discount requires a price diagnosis.** Nothing else is allowed to offer money.
- **No stray discounts** — catches "…and here's 20% off" buried inside
  service-recovery copy.
- **Diagnosed a life change? Then no outreach.** Full stop.
- **No pushy language** — no "act now", no countdowns, no "don't miss out".

---

## Where things live

| You want to change... | Edit |
|---|---|
| A threshold, cooldown, band cut point, seed | `config.py` — *nothing else should hardcode these* |
| What counts as a feature | `features/extract.py` |
| The model itself | `scoring/logistic.py` (implement `RiskScorer` to add a new one) |
| Who gets suppressed | `agent/policy.py` |
| How causes are diagnosed | `agent/stub_llm.py` |
| The wording of messages | `agent/copywriter.py` |
| What the agent isn't allowed to do | `agent/guardrails.py` |

---

## The three things to remember

1. **The LLM never computes the risk number.** It receives it. If you change one
   thing in this codebase, don't change that.
2. **A score is not a decision.** 85% risk with the correct answer being "do
   nothing" is not a bug, it's the point.
3. **The model handles uncertainty; the gate handles certainty.** Anything you
   can *compute* (do they still have product?) belongs in the gate. Anything you
   have to *estimate* belongs in the model.
