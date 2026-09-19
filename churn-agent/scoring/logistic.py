"""Logistic regression risk scorer.

Chosen over gradient boosting on purpose. Three reasons, in order:

1. Attributions are exact. For a linear model the contribution of feature
   j to this customer's log-odds is literally coef_j * z_j. No SHAP
   approximation, no "correlated features make the attribution lie".
   The agent layer is handed a true decomposition.
2. It is well calibrated by construction -- log-loss is a proper scoring
   rule -- and the policy gate thresholds on the probability, so
   calibration is not optional.
3. On a 200-customer cohort, boosting would overfit and we would have to
   spend the demo defending it.

Swap in something stronger by implementing RiskScorer; nothing downstream
of scoring/base.py needs to change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg
from features.extract import FEATURE_LABELS, MODEL_FEATURES
from scoring.base import Attribution, Band, FitReport, RiskResult, RiskScorer, band_for


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """What the retention team actually experiences: they work a top-K list."""
    k = min(k, len(scores))
    if k == 0:
        return 0.0
    top = np.argsort(-scores)[:k]
    return float(y_true[top].mean())


class LogisticRiskScorer(RiskScorer):
    name = "logistic_regression"

    def __init__(self, feature_names: list[str] | None = None) -> None:
        self.feature_names = feature_names or MODEL_FEATURES
        self.pipeline = Pipeline([
            # Undefined rates (see features/extract.py UNDEFINED) arrive as
            # NaN and become the cohort median -- i.e. "no reading", not
            # "zero". The inactivity itself is carried by other features.
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                C=0.25,              # L2; 24 features on 200 rows needs it
                solver="lbfgs",
            )),
        ])
        # Keep DataFrames flowing through the pipeline so the scaler sees
        # feature names and sklearn stops warning about it.
        self.pipeline.set_output(transform="pandas")
        self.report: FitReport | None = None

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> FitReport:
        X = X[self.feature_names]
        y_arr = y.astype(int).to_numpy()

        # Held-out split purely for reporting. We refit on everything
        # afterwards because 200 rows is not enough to throw 30% away.
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_arr, test_size=cfg.TEST_SIZE, random_state=cfg.SEED, stratify=y_arr
        )
        self.pipeline.fit(X_tr, y_tr)
        p_te = self.pipeline.predict_proba(X_te)[:, 1]

        k = min(cfg.PRECISION_AT_K, len(y_te))
        p_at_k = precision_at_k(y_te, p_te, k)
        base = float(y_te.mean())

        report = FitReport(
            model_name=self.name,
            n_train=len(y_tr), n_test=len(y_te), base_rate=base,
            roc_auc=float(roc_auc_score(y_te, p_te)),
            pr_auc=float(average_precision_score(y_te, p_te)),
            brier=float(brier_score_loss(y_te, p_te)),
            precision_at_k=p_at_k, k=k,
            lift_at_k=float(p_at_k / base) if base > 0 else 0.0,
            top_coefficients=[],
        )

        # Refit on the full cohort for the model we actually serve.
        self.pipeline.fit(X, y_arr)
        coefs = self._coefficients()
        report.top_coefficients = sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:10]
        self.report = report
        return report

    # ------------------------------------------------------------------
    def _coefficients(self) -> dict[str, float]:
        clf: LogisticRegression = self.pipeline.named_steps["clf"]
        return dict(zip(self.feature_names, clf.coef_[0]))

    def score(self, customer_id: str, features: dict[str, float]) -> RiskResult:
        row = pd.DataFrame([[features[f] for f in self.feature_names]],
                           columns=self.feature_names)
        probability = float(self.pipeline.predict_proba(row)[0, 1])

        # Exact local decomposition: contribution_j = coef_j * z_j, where
        # z is the standardised value. These sum (with the intercept) to
        # the log-odds -- no approximation anywhere. Run the row through
        # the same impute -> scale steps the model saw.
        imputed = self.pipeline.named_steps["impute"].transform(row)
        z = self.pipeline.named_steps["scale"].transform(imputed).to_numpy()[0]
        coefs = self.pipeline.named_steps["clf"].coef_[0]

        attributions = [
            Attribution(
                feature=name,
                label=FEATURE_LABELS.get(name, name),
                value=float(features[name]),
                contribution=float(coefs[i] * z[i]),
            )
            for i, name in enumerate(self.feature_names)
        ]
        return RiskResult(
            customer_id=customer_id,
            probability=probability,
            band=band_for(probability),
            attributions=attributions,
            model_name=self.name,
        )
