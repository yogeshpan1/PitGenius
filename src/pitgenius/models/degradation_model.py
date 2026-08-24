"""Tire degradation model: predicts clean-lap time as a function of
tire age / compound / circuit / fuel load, with P10/P50/P90 quantile outputs.

Target design (important): we do NOT predict absolute lap times, nor delta
vs the driver's whole-race median. We predict the **delta vs the driver's
per-stint reference lap** — the median clean lap over the first few
tire-life laps of the SAME stint (STINT_REF_MAX_AGE). The original
race-median target conflated tire state with lap context: stint starts
(cool fuel, low traffic) are systematically fast for every tire state,
which produced implausible fresh-tire deltas (~-3 s at tire age 2) and was
a diagnosed cause of the undercut calculator's failed backtest
(DECISIONS.md D22, fix 1). Delta vs the stint's own early-lap reference
isolates within-stint degradation; residual fuel effect is handled by the
fuel_proxy feature (DECISIONS.md D14).

Features:
    tire_life      laps on current set
    compound       SOFT / MEDIUM / HARD (one-hot)
    fuel_proxy     linear fuel-load proxy (DECISIONS.md D14)
    circuit        event identity (LightGBM categorical)

Models: three LightGBM regressors with objective='quantile' at
alpha = 0.1 / 0.5 / 0.9 (DECISIONS.md D6).
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from pitgenius.analysis.degradation import green_flag_laps
from pitgenius.config import MODELS_DIR

FUEL_S_PER_LAP = 0.033
QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}
FEATURES = ["tire_life", "fuel_proxy", "compound", "circuit"]

# Reference window for the per-stint delta target: the median clean lap over
# the first STINT_REF_MAX_AGE tire-life laps of each stint. Small enough to
# be near-fresh, large enough to survive one noisy out-lap.
STINT_REF_MAX_AGE = 3


def build_training_frame(laps: pd.DataFrame) -> pd.DataFrame:
    """Clean laps -> supervised training rows with a per-stint delta target.

    Target: lap_time_s minus the median clean lap time over the first
    STINT_REF_MAX_AGE tire-life laps of the SAME stint (year, round, driver,
    stint_number). Rows whose stint never produced a clean early reference
    lap (very short scraps, heavy timing gaps) have no valid target and are
    dropped — counted and reported honestly rather than silently patched.
    """
    gf = green_flag_laps(laps).copy()
    if gf.empty:
        return gf

    grp = ["year", "round", "driver", "stint_number"]
    # Fuel-adjust lap times before taking the stint reference (DECISIONS.md
    # D14 proxy): otherwise the reference carries the within-stint fuel
    # burn-off (~0.033 s/lap) and old-tire laps look artificially FAST,
    # which verification showed as a falling age-curve.
    gf["_t_adj"] = gf["lap_time_s"] + FUEL_S_PER_LAP * gf["lap_number"]
    ref = (
        gf[gf["tire_life"] <= STINT_REF_MAX_AGE]
        .groupby(grp, as_index=False)["_t_adj"]
        .median()
        .rename(columns={"_t_adj": "stint_ref"})
    )
    n_before = len(gf)
    gf = gf.merge(ref, on=grp, how="left")
    gf["delta"] = gf["_t_adj"] - gf["stint_ref"]
    gf = gf.drop(columns="_t_adj")
    gf = gf.dropna(subset=["delta", "tire_life"])
    gf.attrs["rows_dropped_no_stint_ref"] = n_before - len(gf)

    total = gf.groupby(["year", "round"])["lap_number"].transform("max")
    gf["fuel_proxy"] = FUEL_S_PER_LAP * (total - gf["lap_number"])

    gf["circuit"] = gf["round"].astype(str) + "_" + gf["year"].astype(str)
    # Circuit identity across years: use round number only when the calendar
    # is stable enough; safer to key on (round, year) so each race is its own
    # category — the model then learns per-race baselines from other drivers'
    # laps in the same race, which is exactly the deployment scenario.
    return gf


@dataclass
class DegradationPrediction:
    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray


class TireDegradationModel:
    """Quantile regression on lap-time delta vs tire age."""

    def __init__(self, params: dict | None = None):
        base = dict(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            verbose=-1,
        )
        self.params = {**base, **(params or {})}
        self.models_: dict[str, lgb.LGBMRegressor] = {}

    def _make_xy(self, df: pd.DataFrame):
        X = pd.DataFrame({
            "tire_life": df["tire_life"].astype(float),
            "fuel_proxy": df.get("fuel_proxy",
                                 pd.Series(0.0, index=df.index)),
            "compound": df["tire_compound"].astype("category"),
            "circuit": df["circuit"].astype("category"),
        })
        y = df["delta"]
        return X, y

    def fit(self, train_rows: pd.DataFrame) -> "TireDegradationModel":
        X, y = self._make_xy(train_rows)
        for name, alpha in QUANTILES.items():
            m = lgb.LGBMRegressor(objective="quantile", alpha=alpha,
                                  **self.params)
            m.fit(X, y, categorical_feature=["compound", "circuit"])
            self.models_[name] = m
        return self

    def predict(self, rows: pd.DataFrame) -> DegradationPrediction:
        X, _ = self._make_xy(rows.assign(delta=0.0))
        return DegradationPrediction(
            p10=self.models_["p10"].predict(X),
            p50=self.models_["p50"].predict(X),
            p90=self.models_["p90"].predict(X),
        )

    def predict_p50(self, rows: pd.DataFrame) -> np.ndarray:
        return self.predict(rows).p50

    def save(self, path=None) -> str:
        path = str(path or (MODELS_DIR / "degradation_quantile.joblib"))
        joblib.dump({"models": self.models_, "params": self.params}, path)
        return path

    @classmethod
    def load(cls, path=None) -> "TireDegradationModel":
        path = str(path or (MODELS_DIR / "degradation_quantile.joblib"))
        blob = joblib.load(path)
        obj = cls(params=blob["params"])
        obj.models_ = blob["models"]
        return obj