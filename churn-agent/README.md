# Churn agent — DTC sports nutrition

Spot churn, take a helpful first action. Built for Chatathon 2026, Klaviyo track.

Everything here runs on **synthetic data** with **no API key** and **no network**.
There is no send path anywhere in this project.

---

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # python 3.12 recommended
pip install -r requirements.txt

python run_generate.py        # build the synthetic cohort -> data/out/
python smoke_test.py          # verify the pipeline + the five demo cases
python -m unittest test_dashboard -v  # verify the review workflow
streamlit run app/dashboard.py
```

Headless, if you want to see a trace without the UI:

```bash
python run_pipeline.py                      # cohort summary
python run_pipeline.py --show CUST-0001     # full trace for one customer
```

---

## Using the workspace

The dashboard opens on **Review queue**, with suggested responses ordered by
risk. Each customer shows their risk, a plain-language reason, and a recommended
next step. Search by customer ID, reason, or action, or filter by risk level.

- **Needs review** contains suggested responses that still need a person to review
  them. Open one to see context alongside an editable draft. Save your edits, or
  mark the response reviewed after the draft checks pass.
- **High risk** includes every high-risk customer, including people who should
  not be contacted. **On hold** explains why at-risk customers are being left alone.
- **Reviewed** contains responses reviewed in this browser session. Download
  a labeled demo draft to keep a copy, reopen it for editing, or move to the next
  customer. Marking reviewed never sends a message.

**All customers** includes the entire cohort. **How it works** explains the flow,
links to the five demo cases, and keeps model evaluation available on demand.
Customer details include expandable risk explanations, contact checks,
recommendation reasoning, support history, and recorded signals.

Drafts and review progress live only in Streamlit session state; reloading or
closing the browser can reset them. Edited copy is rechecked for tone, length,
discounts, and the existing response rules before it can be marked reviewed or
downloaded. There is no send path, shared review database, or live integration.

---

## The one architectural rule

Two layers, hard boundary, and the boundary is visible in `pipeline.py`.

```
  events  ──▶  features/extract.py  ──▶  scoring/logistic.py  ──▶  agent/policy.py
                                              │                          │
              DETERMINISTIC LAYER ────────────┴──────────────────────────┘
              pure python + sklearn. no LLM. produces a calibrated
              probability and exact per-feature attributions.
                                                                         │
              AGENT LAYER ───────────────────────────────────────────────┤
              reads unstructured signals, diagnoses WHY, picks an        │
              action, drafts copy. never computes the risk number.       ▼
                                          agent/diagnose.py ──▶ agent/guardrails.py
```

**The LLM never computes the risk number.** It receives the score and the
attributions as input. If you change one thing in this codebase, do not change
that.

The split is not arbitrary. Structured behaviour goes to the model, which is
good at it. Unstructured text — *"I've torn my rotator cuff and I'm out of the
gym for three months"* — goes to the LLM, which is the only layer that can read
it. Neither layer is asked to do the other's job.

---

## Why serving counts

A 30-serving tub at one scoop a day runs out in 30 days. That makes the churn
signal a **physical** question — have they actually run out? — rather than an
arbitrary statistical threshold.

| SKU | servings | per day | days of supply |
|---|---|---|---|
| `WHEY-1LB` | 15 | 1.0 | **15** |
| `WHEY-2LB` | 30 | 1.0 | **30** |
| `WHEY-5LB` | 76 | 1.0 | **76** |
| `CREA-350` | 70 | 1.0 | **70** |
| `PRE-25` | 25 | 1.0 | **25** |
| `ELEC-30` | 30 | 1.5 | **20** |
| `MULTI-60` | 60 | 2.0 | **30** |

A 55-day gap means nothing on a 76-day tub and means everything on a 15-day one.
This falls out of `data/catalog.py` and shows up in three places: as a feature,
as a hard policy rule, and in the drafted copy.

---

## The five demo cases

Seeded by ID, stable across regenerations, asserted in `smoke_test.py`.

| ID | Case | Risk | What happens | Why it matters |
|---|---|---|---|---|
| `CUST-0001` | service failure | 64% MED | **acted** → service recovery | Gap is 2.8× their median and two damaged-shipment tickets sat unanswered for 85 days. Correct action is to fix it, not to discount it. A discount here buys silence. |
| `CUST-0002` | false positive | 36% MED | **suppressed** (still stocked) | 3.06× gap — *worse* than CUST-0001 — but they bought a 5 lb tub and have ~21 days left. Reorders on schedule in the forward window. Naive recency ranks them #99; the supply rule stops them dead. |
| `CUST-0003` | price sensitivity | 49% MED | **acted** → value education | 88% of orders used a promo, browsing without converting, and a ticket asking about subscribe-and-save. Answer is cost-per-serving maths, **not** a percentage off. |
| `CUST-0004` | cooldown | 86% HIGH | **suppressed** (cooldown) | Genuinely at risk and genuinely actionable, but we sent targeted outreach 6 days ago. The gate stops this *before* the agent runs — no tokens spent on someone we cannot contact. |
| `CUST-0005` | legitimate stop | 85% HIGH | **no action** (agent) | Torn rotator cuff, CSAT 5/5, zero complaints. The model says 85% and the model is right — they have churned. Emailing them protein anyway is the tasteless failure mode this whole system exists to avoid. |

CUST-0005 is the one to demo last. It is the case where a correct risk score and
a correct decision point in opposite directions.

---

## How the synthetic data stays honest

Each customer has a **latent state** — `still_training`, `brand_affinity`,
`price_sensitivity`, `goal`, a buy-till-you-die purchase rate and dropout
probability. Every observable is a *noisy* function of it:

```
latent state ──noise──▶ order timing, basket size, session rate,
                        email clicks, ticket sentiment, CSAT
```

Dropout probability rises with `1 - (0.7·affinity + 0.3·satisfaction)`, so a
fading customer leaves a trail before they go — learnable, never readable.

**The firewall.** Latent state is written to `data/out/latent_truth.json`, in a
different format from every feature file, with a warning header. Nothing in
`features/` or `scoring/` imports it. `smoke_test.py` enforces this by **parsing
the AST** of every module in those directories and failing the build on a
forbidden import, a `config.LATENT_TRUTH_FILE` access, or a suspicious string
literal — with docstrings excluded, so the check cannot pass just because a file
talks about the latent state in a comment.

Sanity figure: latent state says 144/200 alive at the cutoff, the observed label
says 47% churned, and **they agree only ~81% of the time**. That gap is
deliberate. The label is a noisy observation of a hidden state, so the model
cannot reverse-engineer the generator.

### The label is personalised, not a fixed window

Scoring a 76-day creatine buyer against the same 90-day window as a 20-day
electrolyte buyer manufactures false churners and teaches the model that large
pack sizes predict churn. That is a label artefact, not a fact about customers.

```
window = clip(1.6 × their own reorder cadence, 60, 180) days
```

Their cadence comes from pre-cutoff data only, so this is not leakage. Windows
range 60–180 days, median 110.

---

## Model

Logistic regression, six features, chosen for a reason:

```
+0.60  reorder_gap_ratio            gap ÷ their own median
-0.50  click_rate_90d               clicks, not MPP-inflated opens
+0.45  days_since_last_session
-0.36  order_count_lifetime
+0.28  promo_order_share
-0.25  sessions_since_last_order
```

PR-AUC **0.825** · ROC-AUC 0.791 · Brier 0.182 · precision@20 **0.80** (1.71×
the 46.7% base rate).

> Figures are for the committed seed (`config.SEED = 20260919`). Regenerate with
> a different seed and they move a few points — `run_pipeline.py` prints the
> live values, and **How it works → Model details & evaluation** in the dashboard
> shows them. Quote those, not these.

**Why only six features when 29 are computed?** With ~94 churn events, fitting
two dozen predictors gives under 4 events per variable against an accepted
floor of 10.
Below that, correlated features trade weight and **their coefficients flip
sign** — the attribution panel starts claiming "more product on hand means more
churn risk". Six predictors is 15.7 events per variable and every sign reads as
common sense. All 29 features are still computed and displayed; the 23 excluded
ones are listed with their reasons in `features/extract.py`.

**Why `days_of_supply_remaining` is not a model feature.** It is the most
important number in the project, and that is exactly why. It is not a
statistical correlate, it is arithmetic: servings ÷ servings-per-day. Asking a
logistic regression to *estimate* a fact we can *compute* is how you get a
wrong-signed coefficient on your headline feature. It is enforced as a hard rule
in the policy gate instead. **The model handles uncertainty; the gate handles
certainty.**

Attributions are exact, not approximated — for a linear model, contribution =
`coefficient × standardised value`, and they sum to the log-odds. The smoke test
verifies this reconstructs to within 1.7e-14. No SHAP needed.

---

## The policy gate

Deterministic, runs **before** the agent, cheapest and most absolute check
first:

1. **consent** — unsubscribed or complained. Permanent, no score overrides it.
2. **cooldown** — targeted outreach within 14 days.
3. **risk band** — LOW never gets contacted.
4. **supply** — still has >7 days of product. Nothing is wrong yet.

Cooldown counts **targeted flow sends only**, not broadcast newsletters. The
brand emails the whole list every ~6 days, so counting broadcasts would suppress
148/200 and make the gate theatre. Counting targeted sends suppresses 49 and
isolates the real cases. This is also how marketing ops actually works.

---

## Swapping in a real LLM

One flag:

```python
# config.py
USE_REAL_LLM = True
LLM_MODEL = "claude-opus-5"
```

`agent/stub_llm.py` (rules, deterministic) and `agent/diagnose.py`
`AnthropicDiagnoser` (Claude via `client.messages.parse()` with a Pydantic
schema) both return `AgentDiagnosis`. The gate, the guardrails, and the
dashboard cannot tell them apart.

Both paths see byte-identical evidence: tool output is prefetched by
`build_request()` rather than fetched through a live tool-call loop, which is
what makes the stub a fair stand-in rather than a mock. `TOOL_SCHEMAS` in
`agent/tools.py` is already in Anthropic tool-use format if you want a real loop
later.

Needs `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or an `ant auth login`
profile. **The stub is the default and the demo does not need a key.**

---

## Guardrails

Run on the agent's *output*. A failure downgrades the action, usually to
`NO_ACTION` — when unsure, send nothing.

- `discount_justified` — a discount requires a price diagnosis. The headline one.
- `no_stray_discount` — catches "…and here's 20% off" buried in service-recovery copy.
- `lifestyle_respect` — diagnosed a life change? Then no outreach, full stop.
- `supply_consistency` — duplicates the gate on purpose, in case the diagnosis ignored the arithmetic.
- `tone` — no urgency, countdowns, or "don't miss out".
- `length`, `message_presence`, `human_approval_required`, `simulation_label`.

---

## File map — who can work on what

Written so the team can split up. Files on different rows do not collide.

| Area | Files | Depends on |
|---|---|---|
| **Tunables** | `config.py` | nothing — everyone reads it |
| **Data contracts** | `data/schema.py` | `config` |
| **Product catalog** | `data/catalog.py` | `schema` |
| **Ticket copy** | `data/text_bank.py` | `schema` — *safe to enrich freely* |
| **Generator** | `data/generator.py` | all of the above |
| **Event access** | `data/store.py` | `config` |
| **Features** | `features/extract.py` | `store`, `catalog` |
| **Scoring** | `scoring/base.py`, `scoring/logistic.py` | `features` |
| **Policy gate** | `agent/policy.py` | `scoring`, `store` |
| **Tools** | `agent/tools.py` | `store`, `policy` |
| **Agent contracts** | `agent/schemas.py` | nothing |
| **Stub brain** | `agent/stub_llm.py` | `schemas`, `copywriter` |
| **Message copy** | `agent/copywriter.py` | `schemas` — *safe to iterate freely* |
| **Real LLM** | `agent/diagnose.py` | `schemas`, `tools` |
| **Guardrails** | `agent/guardrails.py` | `schemas` |
| **Orchestration** | `pipeline.py` | everything |
| **Design tokens** | `app/theme.py` | nothing — *safe to restyle freely* |
| **UI** | `app/dashboard.py` | `pipeline`, `theme` |
| **Review workflow** | `app/workflow.py` | `pipeline`, `guardrails` |

`.streamlit/config.toml` is committed deliberately. Without `headless = true`,
`streamlit run` stops on a first-run "enter your email" prompt and never starts
the server — not something to discover ten minutes before a demo. It also
carries the theme, which matters more than it sounds: at Streamlit's default
`primaryColor` every filter chip renders bright red, which reads as *error* on
what is only a filter.

Five files beyond the original spec, each for a stated reason:
`data/store.py` (feature extraction stays pure — the store owns I/O and enforces
the temporal cutoff), `data/text_bank.py` and `agent/copywriter.py` (the two
files people iterate on most, isolated so they need no knowledge of the logic
around them), and `agent/schemas.py` (so `stub_llm.py` and `diagnose.py` can be
edited in parallel without conflicting), and `app/theme.py` (colour and
spacing tokens, so restyling never means touching layout logic).

### Interface hierarchy

Warm neutral surfaces and a restrained green accent keep the workspace calm.
Risk badges pair color with a percentage and text label. The queue keeps risk,
reason, recommended response, and review status distinct. Detailed model and
policy information is available in expandable sections instead of competing
with the next action. The layout adapts to the available content width.

---

## Honest caveats — say these before a judge does

- **All numbers are on synthetic data with a known generator.** PR-AUC 0.825 is
  ground-truth recovery, not a claim about real customers.
- **No uplift modelling.** We rank who is at risk, not who is *saveable*. The
  research says persuadables-vs-sleeping-dogs is the real differentiator and we
  have not built it. `scoring/base.py` is the seam where it would go.
- **No BTYD or survival model.** `RiskScorer` exists so P(alive) from a
  BG/NBD fit can drop in behind the same interface. Not done.
- **Single train/test split on 200 rows.** The reported metrics move a few
  points on a different seed. No cross-validation, no confidence intervals.
- **Calibration is asserted, not measured.** Logistic regression optimises a
  proper scoring rule so it is well-calibrated by construction, and Brier 0.182
  is consistent with that — but there is no reliability diagram.
  `config.CALIBRATE` is a stub; isotonic would need more than 200 points.
- **The stub is rules, not intelligence.** It is faithful about *shape*, not
  about judgement. Flip `USE_REAL_LLM` to see the difference.
- **The label is still a heuristic**, personalised window or not. A customer who
  buys elsewhere for 100 days and returns on day 200 is labelled churned.

## Deliberately not built

No uplift modelling. No BTYD/survival. No gradient boosting. No auth, no
database, no Docker, no deployment config. Pipeline checks live in `smoke_test.py`
and review interaction tests in `test_dashboard.py`. No
real API integration on the default path. **No send functionality of any kind.**
