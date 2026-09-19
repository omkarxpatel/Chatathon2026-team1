#!/usr/bin/env python3
"""The one test. Run it before you demo:  python smoke_test.py

Three things it checks, in order of how embarrassing they would be on stage:

  1. LEAKAGE FIREWALL -- nothing in features/ or scoring/ can reach the
     latent state or the churn label. Checked by parsing the AST, not by
     grepping for the word "latent", because those files *discuss* the
     latent state in comments and a text search would pass for the wrong
     reason.
  2. NO SEND PATH -- nothing anywhere opens a network or mail connection.
  3. PIPELINE + ARCHETYPES -- the whole thing runs, and the five seeded
     demo customers still produce the outcomes the demo script claims.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

FAILURES: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {label}")
    else:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL  {label}  {detail}")


# ==========================================================================
# 1. Leakage firewall
# ==========================================================================
# Names that would mean the scoring path can see the answer.
FORBIDDEN_IMPORTS = {"LatentState"}
FORBIDDEN_ATTRS = {"LATENT_TRUTH_FILE", "LABELS_FILE"}
FORBIDDEN_STRINGS = {"latent_truth", "latent_truth.json", "alive_at_cutoff",
                     "true_dropout_date", "still_training", "brand_affinity"}
SCORING_PATH_DIRS = ["features", "scoring"]


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Docstrings are ast.Constant too. These files *discuss* the latent
    state at length in their docstrings -- that is the documentation
    doing its job, not a leak. Exclude them from the string scan."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                found.add(id(body[0].value))
    return found


def audit_module(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    docstrings = _docstring_nodes(tree)
    problems = []

    for node in ast.walk(tree):
        # `from data.schema import LatentState`
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORTS:
                    problems.append(f"imports {alias.name} (line {node.lineno})")
            if node.module and "latent" in node.module.lower():
                problems.append(f"imports from {node.module} (line {node.lineno})")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "latent" in alias.name.lower():
                    problems.append(f"imports {alias.name} (line {node.lineno})")

        # `cfg.LATENT_TRUTH_FILE`
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            problems.append(f"reads config.{node.attr} (line {node.lineno})")

        # a hardcoded path or column name (docstrings excluded above)
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings):
            for needle in FORBIDDEN_STRINGS:
                if needle in node.value:
                    problems.append(
                        f"string literal contains {needle!r} (line {node.lineno})"
                    )
    return problems


print("\n[1] Leakage firewall -- can the scoring path see the answer?")
for directory in SCORING_PATH_DIRS:
    for path in sorted((ROOT / directory).glob("*.py")):
        problems = audit_module(path)
        check(not problems, f"{directory}/{path.name} is clean", "; ".join(problems))

# The held-out file must actually exist and actually be separate.
import config as cfg  # noqa: E402

check(
    cfg.LATENT_TRUTH_FILE.exists(),
    "latent_truth.json was written",
    "run `python run_generate.py` first",
)
check(
    cfg.LATENT_TRUTH_FILE.suffix == ".json"
    and all(f.suffix == ".parquet" for f in [cfg.CUSTOMERS_FILE, cfg.ORDERS_FILE]),
    "ground truth is stored apart from the feature data",
)

# ==========================================================================
# 2. No send path
# ==========================================================================
print("\n[2] No send path -- can this thing email anybody?")
NETWORK_MARKERS = {"smtplib", "sendgrid", "boto3", "twilio", "mailgun", "postmark"}
offenders = []
for path in sorted(ROOT.rglob("*.py")):
    if ".venv" in path.parts:
        continue
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module.split(".")[0]]
        for name in names:
            if name in NETWORK_MARKERS:
                offenders.append(f"{path.relative_to(ROOT)} imports {name}")
check(not offenders, "no mail or messaging client is imported anywhere",
      "; ".join(offenders))

# ==========================================================================
# 3. Pipeline end to end + the five demo archetypes
# ==========================================================================
print("\n[3] Pipeline -- does the whole thing run, and do the demos still hold?")

from agent.schemas import Action, Cause  # noqa: E402
from pipeline import run_cohort  # noqa: E402

run = run_cohort()
results = run.by_id()

check(len(run.results) == cfg.N_CUSTOMERS,
      f"scored all {cfg.N_CUSTOMERS} customers", f"got {len(run.results)}")
check(0.0 < run.fit.pr_auc <= 1.0, f"model fit produced metrics "
      f"(PR-AUC {run.fit.pr_auc:.3f})")
check(all(0.0 <= r.risk.probability <= 1.0 for r in run.results),
      "every probability is in [0, 1]")

# Attributions must reconstruct the log-odds exactly, or the drill-down
# is showing numbers that do not add up to the score above them. For a
# linear model, logit(p) - sum(contributions) is the intercept, so it has
# to be the SAME constant for every customer.
import math  # noqa: E402

implied = [
    math.log(r.risk.probability / (1 - r.risk.probability))
    - sum(a.contribution for a in r.risk.attributions)
    for r in run.results
    if 1e-9 < r.risk.probability < 1 - 1e-9
]
spread = max(implied) - min(implied)
check(spread < 1e-6,
      f"attributions reconstruct the log-odds exactly (intercept "
      f"{implied[0]:+.4f}, spread {spread:.2e})",
      f"implied intercept varies by {spread:.2e} across customers")

EXPECTED = {
    "CUST-0001": ("service_recovery", Cause.SERVICE_FAILURE, Action.SERVICE_RECOVERY,
                  "eligible"),
    "CUST-0002": ("false_positive_stocked", None, Action.NO_ACTION, "suppressed"),
    "CUST-0003": ("price_sensitive", Cause.PRICE_SENSITIVITY, Action.VALUE_EDUCATION,
                  "eligible"),
    "CUST-0004": ("cooldown_suppressed", None, Action.NO_ACTION, "suppressed"),
    "CUST-0005": ("legitimate_stop", Cause.LIFESTYLE_CHANGE, Action.NO_ACTION,
                  "eligible"),
}

for cid, (archetype, cause, action, gate) in EXPECTED.items():
    r = results.get(cid)
    if r is None:
        check(False, f"{cid} exists", "missing from the cohort")
        continue
    ok = (
        r.archetype == archetype
        and r.final_action == action
        and (r.policy.eligible if gate == "eligible" else not r.policy.eligible)
        and (cause is None or (r.diagnosis and r.diagnosis.cause == cause))
    )
    got = (
        f"archetype={r.archetype}, gate={r.policy.outcome.value}, "
        f"cause={r.diagnosis.cause.value if r.diagnosis else '-'}, "
        f"action={r.final_action.value}"
    )
    check(ok, f"{cid} ({archetype}) behaves as the demo claims", got)

# Never discount without a price diagnosis -- the headline guardrail.
bad_discounts = [
    r.customer_id for r in run.results
    if r.final_action == Action.DISCOUNT_OFFER
    and (not r.diagnosis or r.diagnosis.cause != Cause.PRICE_SENSITIVITY)
]
check(not bad_discounts, "no discount was proposed without a price diagnosis",
      str(bad_discounts))

# Every drafted message must be flagged for a human.
unapproved = [
    r.customer_id for r in run.results
    if r.diagnosis and r.diagnosis.has_message and not r.diagnosis.requires_human_approval
]
check(not unapproved, "every drafted message requires human approval", str(unapproved))

# ==========================================================================
print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("All smoke checks passed.")
