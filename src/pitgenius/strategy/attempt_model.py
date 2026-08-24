"""P(attempt) model — DECISIONS.md D22 fix 2.

The undercut calculator answers P(flip | an adjacent-stop interaction
happens). Historically it was only ever scored on interactions teams CHOSE
to attempt, and teams attempt precisely when conditions already favour them
(detected-attempt base rates: 96% undercut, 2% overcut). That selection bias
made the calculator look anti-correlated on the detected sample.

This module models the OTHER half of the deployment question separately:

    P(responder follows within the attempt window | pair features)

so that P(success | attempt) is never again estimated on a self-selected
sample, and a deployment-facing combined probability can be decomposed as

    P(response AND flip) = P(attempt) * P(flip | attempt)

The two factors are trained and reported independently and are NEVER merged
into a fake "success probability" for non-attempted pairs (on those pairs
the outcome is not governed by flip dynamics at all).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Features available BEFORE either car commits (no outcome leakage).
NUMERIC_FEATURES = [
    "gap_at_attempt_s",   # running-order gap proxy before the first stop
    "attacker_age",       # first pitter's tire age at its stop lap
    "defender_age",       # responder's tire age at its stop lap
    "age_diff",           # defender_age - attacker_age
    "same_compound",      # 1 if both on the same compound
]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["gap_at_attempt_s"] = pd.to_numeric(
        df["gap_at_attempt_s"], errors="coerce")
    out["attacker_age"] = pd.to_numeric(df["attacker_age"], errors="coerce")
    out["defender_age"] = pd.to_numeric(df["defender_age"], errors="coerce")
    out["age_diff"] = out["defender_age"] - out["attacker_age"]
    out["same_compound"] = (
        df["attacker_compound"].astype(str)
        == df["defender_compound"].astype(str)).astype(float)
    return out[NUMERIC_FEATURES]


class AttemptModel:
    """Logistic P(responder follows within the attempt window)."""

    def __init__(self, C: float = 1.0):
        self.C = C
        self.clf_ = LogisticRegression(max_iter=2000, C=C)
        self.base_rate_: float | None = None

    def fit(self, pairs: pd.DataFrame) -> "AttemptModel":
        y = pairs["is_attempt"].astype(int)
        self.base_rate_ = float(y.mean())
        self.clf_.fit(build_feature_matrix(pairs), y)
        return self

    def predict_proba(self, pairs: pd.DataFrame) -> np.ndarray:
        return self.clf_.predict_proba(build_feature_matrix(pairs))[:, 1]
