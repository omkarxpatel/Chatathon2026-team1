"""Every tunable in one place.

Nothing else in the project should hardcode a threshold, a date, or a seed.
If you find yourself typing a magic number elsewhere, put it here instead.
"""

from datetime import date, timedelta
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_OUT = ROOT / "data" / "out"

# Observable event streams -- these are the ONLY files the feature extractor
# is allowed to read.
CUSTOMERS_FILE = DATA_OUT / "customers.parquet"
ORDERS_FILE = DATA_OUT / "orders.parquet"
ORDER_LINES_FILE = DATA_OUT / "order_lines.parquet"
SESSIONS_FILE = DATA_OUT / "web_sessions.parquet"
MESSAGES_FILE = DATA_OUT / "message_events.parquet"
TICKETS_FILE = DATA_OUT / "support_tickets.parquet"
LABELS_FILE = DATA_OUT / "labels.parquet"

# Held-out ground truth. Written by the generator, read by nobody in
# features/ or scoring/. See smoke_test.py, which enforces this.
LATENT_TRUTH_FILE = DATA_OUT / "latent_truth.json"

# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
SEED = 20260919

N_CUSTOMERS = 200
HISTORY_DAYS = 240  # ~8 months of observable history before the cutoff

# Timeline, anchored on the end of the simulation and worked backwards:
#
#   SIM_START ------------ HISTORY_DAYS ------------ AS_OF_DATE
#                      (features come from here)          |
#                                       AS_OF_DATE -- LABEL_WINDOW_DAYS -- SIM_END
#                                            (the label comes from here)
#
# Features are computed strictly BEFORE AS_OF_DATE; the churn label is
# observed strictly AFTER it. This is the temporal-leakage firewall.
#
# The label window is PERSONALISED, not fixed. A customer on a 76-day
# creatine cycle who has not reordered in 90 days has not churned -- they
# are not due yet. Scoring them against the same 90-day window as a
# 20-day electrolyte buyer manufactures false churners and teaches the
# model that big pack sizes predict churn, which is a label artefact
# rather than a fact about customers.
#
#   window = clip(LABEL_CYCLE_MULTIPLE x their own cycle, MIN, MAX)
#
# Their "own cycle" is their median inter-purchase interval, falling back
# to the catalog days-of-supply of their last basket. Both are computed
# from data strictly BEFORE the cutoff, so this is not leakage.
LABEL_CYCLE_MULTIPLE = 1.6
LABEL_WINDOW_MIN_DAYS = 60
LABEL_WINDOW_MAX_DAYS = 180

# How far past the cutoff we simulate, so even the longest personalised
# window has real data behind it.
LABEL_HORIZON_DAYS = 200

AS_OF_DATE = date(2026, 6, 17)
SIM_END = AS_OF_DATE + timedelta(days=LABEL_HORIZON_DAYS)
SIM_START = AS_OF_DATE - timedelta(days=HISTORY_DAYS)

# Customers sign up across a wide range so tenure varies realistically.
# Events before SIM_START are simulated but not recorded.
SIGNUP_EARLIEST = SIM_START - timedelta(days=300)
SIGNUP_LATEST = SIM_START + timedelta(days=45)

# Latent-state population parameters.
P_TRAINING_STOPS = 0.18        # chance a customer stops training mid-history
P_GOAL_CHANGES = 0.22          # chance goal shifts (changes their basket)
BRAND_AFFINITY_DRIFT = 0.15    # std-dev of affinity drift over full history

# Behaviour noise. Higher = observables are a weaker readout of latent state.
# Keep these non-trivial: the point is that the model must infer, not read off.
ORDER_TIMING_NOISE = 0.30      # lognormal sigma on inter-purchase interval
SESSION_RATE_NOISE = 0.45
EMAIL_ENGAGEMENT_NOISE = 0.20
TICKET_SENTIMENT_NOISE = 0.25

# --------------------------------------------------------------------------
# Risk bands (mirrors Klaviyo's published Low/Medium/High cut points)
# --------------------------------------------------------------------------
BAND_LOW_MAX = 0.33
BAND_MEDIUM_MAX = 0.66

# --------------------------------------------------------------------------
# Policy gate (deterministic, runs BEFORE the agent)
# --------------------------------------------------------------------------
COOLDOWN_DAYS = 14             # no outreach within N days of the last send
MIN_BAND_TO_ACT = "MEDIUM"     # LOW-band customers are never contacted
SUPPLY_BUFFER_DAYS = 7         # still >N days of product on hand => leave alone

# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
TEST_SIZE = 0.30
PRECISION_AT_K = 20            # top-K list the retention team would actually work

# Logistic regression is well-calibrated by construction (log-loss is a proper
# scoring rule). Platt/isotonic wrapping is available but needs more data than
# a 200-customer cohort provides -- see README.
CALIBRATE = False

# --------------------------------------------------------------------------
# Agent layer
# --------------------------------------------------------------------------
# Which brain drives the agent loop in agent/loop.py. Both call the same
# tools, produce the same AgentRun, and render identically in the UI.
# False  -> agent/rules_agent.py   (deterministic, no API key, no network)
# True   -> agent/claude_agent.py  (a real Claude tool-use loop)
USE_REAL_LLM = False
LLM_MODEL = "claude-opus-5"
LLM_MAX_TOKENS = 2000

# Guardrails
MAX_MESSAGE_WORDS = 130
BANNED_PUSHY_PHRASES = [
    "act now", "last chance", "don't miss out", "dont miss out",
    "limited time", "hurry", "expires today", "final call",
    "only hours left", "buy now", "!!!", "urgent",
]

# Every generated message carries this. There is no send path in this project.
SIMULATION_BANNER = "[SIMULATED -- NOT SENT]"

# --------------------------------------------------------------------------
# Demo archetypes. Seeded by ID so they are stable across regenerations.
# --------------------------------------------------------------------------
ARCHETYPE_IDS = {
    "CUST-0001": "service_recovery",       # true positive, diagnosable cause
    "CUST-0002": "false_positive_stocked", # long gap but bought a 5 lb tub
    "CUST-0003": "price_sensitive",        # browsing, not converting, promo-led
    "CUST-0004": "cooldown_suppressed",    # at risk, but contacted 6 days ago
    "CUST-0005": "legitimate_stop",        # injury, happy customer, leave alone
}
