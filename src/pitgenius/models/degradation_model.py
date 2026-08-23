"""Tire degradation model: predicts clean-lap time as a function of
tire age / compound / circuit / fuel load, with P10/P50/P90 quantile outputs.

Target design (important): we do NOT predict absolute lap times. We predict
the **delta vs the driver's own median clean lap in that race**. This removes
car-performance and driver-skill differences, which are not what a strategy
model should be learning; strategy decisions depend on relative degradation.

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


def build_training_frame(laps: pd.DataFrame) -> pd.DataFrame:
    """Clean laps -> supervised training rows with delta target."""
    gf = green_flag_laps(laps).copy()
    if gf.empty:
        return gf

    med = gf.groupby(["year", "round", "driver"])["lap_time_s"].transform("median")
    gf["delta"] = gf["lap_time_s"] - med

    total = gf.groupby(["year", "round"])["lap_number"].transform("max")
    gf["fuel_proxy"] = FUEL_S_PER_LAP * (total - gf["lap_number"])

    gf["circuit"] = gf["round"].astype(str) + "_" + gf["year"].astype(str)
    # Circuit identity across years: use round number only when the calendar
    # is stable enough; safer to key on (round, year) so each race is its own
    # category — the model then learns per-race baselines from other drivers'
    # laps in the same race, which is exactly the deployment scenario.
    gf = gf.dropna(subset=["delta", "tire_life"])
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
            "tire_life": df["tire_life"],
            "fuel_proxy": df.get("fuel_proxy",
                                 pd.Series(0.0, index=df.index)),
            "compound": df["tire_compound"],
            "circuit": df["circuit"],
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