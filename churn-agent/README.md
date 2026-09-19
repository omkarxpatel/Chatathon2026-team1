# Cadence — DTC sports nutrition

A retention agent. A model ranks who is likely to stop buying; the agent then
investigates the ones worth investigating — reading their support tickets,
their orders, the arithmetic on whether they have actually run out — works out
*why*, and either drafts a response or says plainly that we should leave them
alone. A person reviews everything.

Built for Chatathon 2026, Klaviyo track.

Everything here runs on **synthetic data** with **no API key** and **no network**.
There is no send path anywhere in this project.

---

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # python 3.12 recommended
pip install -r requirements.txt

python run_generate.py        # build the synthetic cohort -> data/out/
python smoke_test.py          # pipeline, the five demo cases, the agent loop
python -m unittest test_agent_loop test_dashboard -v   # loop edge cases + review workflow
streamlit run app/dashboard.py
```

Headless, if you want to see a trace without the UI:

```bash
python run_pipeline.py                      # cohort summary, tool usage, drifting list
python run_pipeline.py --show CUST-0001     # the agent's whole run, risk curve included
```

---

## Using the workspace

The dashboard opens on **Review queue**, with suggested responses ordered by
risk. The strip across the top is the agent's last run as four numbers —
customers scanned, customers it looked at closely, tool calls it chose to make,
drafts waiting for you. Each row shows risk, a plain-language reason, and a
recommended next step. Search by customer ID, reason, or action, or filter by
risk level.

Opening a customer leads with the recommendation, then **How the agent worked
this out**: every tool call it made, the sentence it wrote before making it, and
the finding in plain language. Raw tool output is one expander down. That
timeline is the explanation — not a summary of one.

- **Needs review** contains suggested responses that still need a person to review
  them. Open one to see context alongside an editable draft. Save your edits, or
  mark the response reviewed after the draft checks pass.
- **High risk** includes every high-risk customer, including people who should
  not be contacted. **On hold** explains why at-risk customers are being left alone.
- **Drifting** holds customers who are not high risk yet but whose risk is
  climbing the way it climbed for people who left. Nothing else surfaces them.
- **Reviewed** contains responses reviewed in this browser session. Download
  a labeled demo draft to keep a copy, reopen it for editing, or move to the next
  customer. Marking reviewed never sends a message.

**All customers** includes the entire cohort. **Flow map** is the picture with no
prose to read: five bands from raw events, through the deterministic score and
gate, through the tool calls the agent chose to make, to the person at the end.
Every number on it is read off the loaded run. **How it works** walks the same
flow in words, lists every tool the agent can reach for and how often it used
each one, links to the five demo cases, and keeps model evaluation behind a
disclosure. Customer details include expandable risk explanations, contact
checks, safety checks on the draft, support history, and recorded signals.

Drafts and review progress live only in Streamlit session state; reloading or
closing the browser can reset them. Edited copy is rechecked for tone, length,
discounts, and the existing response rules before it can be marked reviewed or
downloaded. There is no send path, shared review database, or live integration.

---

## The flow, end to end

```
  raw events ──▶ features ──▶ risk model ──▶ policy gate ──▶ AGENT LOOP ──▶ guardrails ──▶ you
  orders          29 signals   logistic       4 cheap rules   Claude or       9 checks      review
  sessions        per person   regression     run first       the rule        on the        and edit
  emails                           │                          brain, with     output
  tickets                          │  ── DETERMINISTIC ──     tools
                                   └──▶ trajectory: the same model rescored at 13
                                        earlier dates. which direction, not how bad.
```

**The LLM never computes the risk number.** It receives the score and the
attributions as input. If you change one thing in this codebase, do not change
that.

The split is not arbitrary. Structured behaviour goes to the model, which is
good at it. Unstructured text — *"I've torn my rotator cuff and I'm out of the
gym for three months"* — goes to the LLM, which is the only layer that can read
it. Neither layer is asked to do the other's job.

The whole thing is visible in order in `pipeline.py`.

---

## The agent loop

`agent/loop.py` is twenty lines and it is the agentic part:

```
  brief in  ─▶  [ think ─▶ call a tool ─▶ read the result ] × N  ─▶  decide
```

The brief is the score, the band, and the model's top attributions. **No
evidence.** The agent has to go and get that itself, which is what makes the
investigation on screen a real one rather than a re-narration of a prefetched
blob — and what lets it stop after one tool call when that call already answers
the question.

```
$ python run_pipeline.py --show CUST-0001

[3] AGENT  rules  --  5 steps, stopped because it decided

      1. Before reading anything into the gap, work out whether they have actually run out.
         -> check_product_supply()
         =  Ran out about 20 days ago (Creatine Monohydrate, 350 g)

      2. They are out of product. Read their support history to see whether we
         caused this, or whether they told us something about themselves.
         -> read_support_tickets()
         =  2 support tickets, 2 still open — oldest is 85 days old

      3. An unanswered complaint about a damaged shipment, 85 days old: "Second tub
         in a row arrived with the seal broken..." — pull the orders so the reply
         names the one that went wrong.
         -> read_order_history()
         =  7 orders, last one 74 days ago — 1 refunded or cancelled

      4. Before writing anything, confirm we are still allowed to contact them.
         -> check_contact_policy()
         =  No targeted outreach ever sent

      5. They did not drift away — we broke something and then did not answer.
         -> propose_outreach(cause='service_failure', action='service_recovery', ...)
```

### The tools

Two kinds, and the split is the design. Five **investigate** tools read one
customer; three **decide** tools end the run. The agent cannot dribble out a
conclusion in prose — it has to commit to a structured tool call the guardrails
can then check.

| | Tool | What it does |
|---|---|---|
| investigate | `check_product_supply` | servings ÷ servings-per-day. Call it first: a long gap on a 76-day tub is arithmetic, not disengagement |
| investigate | `read_support_tickets` | full text, unsummarised. The signal the whole agent layer exists to read |
| investigate | `read_order_history` | what, when, how much, and whether a promo was used |
| investigate | `read_engagement` | clicks and sessions. Opens are reported and flagged as MPP-inflated |
| investigate | `check_contact_policy` | opt-out status and cooldown. Duplicates the gate on purpose |
| **decide** | `propose_outreach` | an action that fits the cause, plus the drafted words |
| **decide** | `recommend_no_contact` | a first-class outcome, not a failure |
| **decide** | `escalate_to_human` | past what a template should touch |

Every tool is pre-scoped to the customer under review. There is no argument
anywhere that lets the agent ask about somebody else.

### Three safety properties, enforced by the loop, not trusted to the brain

- **bounded** — `MAX_STEPS` tool calls, then it stops
- **error-tolerant** — a bad tool call is an error handed back, not a crash
- **fail-quiet** — running out of steps produces `NO_ACTION`, never a guess

`smoke_test.py` asserts all of this on every run: that the agent never ran on a
customer the gate stopped, that every run ended by calling a decision tool, that
every run fetched evidence first, and that nothing exceeded the step limit.

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

## Direction of travel

The score answers *how bad*. It cannot answer *which way*, and those are
different questions:

| | 6 months ago | today | band |
|---|---|---|---|
| Customer A | 55% | 55% | MEDIUM |
| Customer B | 22% | 55% | MEDIUM |

A has always reordered slowly and is fine. B is falling off a cliff and has not
landed yet. Same band, same queue position, opposite situations.

`features/trajectory.py` **rewinds the clock**: it rebuilds each customer's
feature vector at 13 dates across 180 days and runs *the same fitted model* over
each one. That gives a risk curve, and its recency-weighted slope is `momentum`.

**No second model, and no trend features in the scorer.** We measured that
version and it is worse — PR-AUC 0.755 → 0.742. Ninety-four churn events cannot
carry thirteen predictors, the same arithmetic that keeps the feature list at
six. The trajectory is a parallel signal; the risk number is untouched.

Out of fold, 5 seeds, against a 47% base rate:

| Group | Churn rate |
|---|---|
| CLIMBING | **76%** |
| RECOVERING | 30% |
| **Below HIGH band and climbing** — the "Drifting" queue | **61.5% ± 5.7** |
| Below HIGH band, not climbing | 30.0% |

**2.05×** over the customers sitting beside them, none of whom the workspace
surfaces today.

Two things make this honest rather than a demo trick:

- **Rewinding cannot leak.** `store.events_for(cid, as_of)` truncates, so a
  rewound snapshot physically cannot see past its own cutoff. Same firewall as
  the live features, reused. `smoke_test.py` asserts it.
- **The common-mode bias is removed.** `order_count_lifetime` is cumulative, so
  every rewound snapshot looks riskier than it deserves and every curve tilts
  down by roughly 4.7 pts/100d. That is an artefact of looking backwards, not a
  recovering cohort, so momentum is centred on the cohort median before anything
  is classified.

**A climbing trajectory does not open the policy gate.** It is a reason to look,
never consent to contact. Letting a trend override the gate would email people
with a full tub in the cupboard — the exact failure `CUST-0002` exists to
prevent. Caveats and the full table are in `SCORING.md`.

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

Two brains, one protocol (`Brain` in `agent/loop.py` — `reset`, `next_move`,
`observe`, `observe_error`):

| | |
|---|---|
| `agent/rules_agent.py` | `RuleBrain`. Deterministic cascade, no API key, no network. |
| `agent/claude_agent.py` | `ClaudeBrain`. A real Anthropic tool-use loop over `TOOL_SCHEMAS`. |

Both drive the *same* loop over the *same* tools and produce the same
`AgentRun`, so the gate, the guardrails, the CLI and the dashboard cannot tell
them apart — and neither can the demo. That is what makes the rule brain a fair
stand-in rather than a mock: it is not a different code path, it is a different
answer to "what should I call next?".

`ClaudeBrain` keeps a message list, appends each `tool_result`, and is reset per
customer — carrying one customer's tickets into the next customer's reasoning
would be a privacy bug, not a feature.

Needs `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or an `ant auth login`
profile. **The rule brain is the default and the demo does not need a key.**

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
| **Trajectory** | `features/trajectory.py` | `extract`, `store` |
| **Scoring** | `scoring/base.py`, `scoring/logistic.py` | `features` |
| **Policy gate** | `agent/policy.py` | `scoring`, `store` |
| **Agent contracts** | `agent/schemas.py` | nothing |
| **Tools** | `agent/tools.py` | `store`, `policy`, `schemas` |
| **The loop** | `agent/loop.py` | `tools`, `schemas` |
| **Rule brain** | `agent/rules_agent.py` | `loop`, `copywriter` |
| **Claude brain** | `agent/claude_agent.py` | `loop`, `tools` |
| **Message copy** | `agent/copywriter.py` | `schemas` — *safe to iterate freely* |
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

Six files beyond the original spec, each for a stated reason:
`data/store.py` (feature extraction stays pure — the store owns I/O and enforces
the temporal cutoff), `data/text_bank.py` and `agent/copywriter.py` (the two
files people iterate on most, isolated so they need no knowledge of the logic
around them), `agent/schemas.py` (so the two brains can be edited in parallel
without conflicting), `agent/loop.py` (the loop is the thing both brains share,
so it cannot live inside either), and `app/theme.py` (colour and spacing tokens,
so restyling never means touching layout logic).

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
- **The default brain is rules, not intelligence.** It runs a real loop and
  makes real tool calls, but the judgement behind "what should I check next?"
  is a cascade, not a model. It is faithful about *shape*, not about judgement.
  Flip `USE_REAL_LLM` to see the difference.
- **The trajectory is correlated with the score** (r ≈ 0.63–0.69), not
  independent evidence — it is a second read on the same signals. It finds
  nothing in the LOW band; the lift is real in MEDIUM and marginal in HIGH.
- **The label is still a heuristic**, personalised window or not. A customer who
  buys elsewhere for 100 days and returns on day 200 is labelled churned.

## Deliberately not built

No uplift modelling. No BTYD/survival. No gradient boosting. No auth, no
database, no Docker, no deployment config. No multi-agent anything, no memory
across customers, and no planner — one loop, eight tools, a step limit. Pipeline
and loop checks live in `smoke_test.py`, loop edge cases in `test_agent_loop.py`,
and review interaction tests in `test_dashboard.py`. No real API integration on
the default path. **No send functionality of any kind.**
