"""Tire degradation analysis: lap time vs tire age, by compound and circuit."""
from __future__ import annotations

import numpy as np
import pandas as pd


def green_flag_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Filter to green-flag, non-pit laps with valid lap times.

    TrackStatus '1' = all clear. FastF1 concatenates flags per lap
    (e.g. '45' means SC then clear); we keep only laps whose status is
    exactly '1' to avoid SC-contaminated pace.
    """
    df = laps.copy()
    mask = (
        df["lap_time_s"].notna()
        & (df["is_pit_out_lap"] == 0)
        & (df["track_status"].fillna("0") == "1")
        & df["tire_compound"].isin(["SOFT", "MEDIUM", "HARD"])
    )
    return df[mask]


def normalize_to_fuel_corrected(laps: pd.DataFrame) -> pd.DataFrame:
    """Add fuel-correction column: F1 cars burn ~0.033 s/lap of fuel effect.

    We do NOT attempt true fuel correction (fuel mass per lap is not public);
    instead we add a linear fuel-load proxy term used consistently across all
    models: expected lap time decreases ~0.033s per lap from race start.
    See DECISIONS.md D14.
    """
    df = laps.copy()
    total = df.groupby(["year", "round"])["lap_number"].transform("max")
    df["fuel_proxy"] = 0.033 * (total - df["lap_number"])
    return df


def stint_summary(laps: pd.DataFrame) -> pd.DataFrame:
    """Per-stint aggregate: mean clean lap time vs tire age curve points."""
    gf = green_flag_laps(laps)
    return (
        gf.groupby(["year", "round", "driver", "stint_number",
                    "tire_compound"], as_index=False)
        .agg(
            stint_len=("lap_number", "count"),
            max_tire_age=("tire_life", "max"),
            mean_lap=("lap_time_s", "mean"),
            best_lap=("lap_time_s", "min"),
        )
    )


def degradation_curve(laps: pd.DataFrame, circuit_round: tuple[int, int] | None,
                      compound: str, min_stint_len: int = 5) -> pd.DataFrame:
    """Mean lap time by tire age for one circuit+compound, pooled drivers.

    Lap times are normalized per-driver-per-race by subtracting each driver's
    median clean lap in that race, so the curve shows *delta* seconds vs tire
    age rather than absolute pace differences between cars.
    """
    gf = green_flag_laps(laps)
    if circuit_round is not None:
        year, round_ = circuit_round
        gf = gf[(gf["year"] == year) & (gf["round"] == round_)]
    gf = gf[gf["tire_compound"] == compound]

    # Per driver-race normalization
    med = gf.groupby(["year", "round", "driver"])["lap_time_s"].transform("median")
    gf = gf.assign(delta=gf["lap_time_s"] - med)

    # Drop stints shorter than min_stint_len (out-lap noise, short scraps)
    stint_len = gf.groupby(["year", "round", "driver", "stint_number"])[
        "lap_number"].transform("count")
    gf = gf[stint_len >= min_stint_len]

    return (
        gf.groupby("tire_life", as_index=False)
        .agg(mean_delta=("delta", "mean"),
             n_obs=("delta", "count"))
        .sort_values("tire_life")
    )


def fit_linear_degradation(curve: pd.DataFrame, max_age: int | None = None):
    """Fit delta ~ a * tire_age + b on the age window [2, max_age].

    Returns (slope_sec_per_lap, intercept, r_squared). Slope is the headline
    "degradation rate" number shown in the dashboard.
    """
    c = curve[curve["tire_life"] >= 2]
    if max_age:
        c = c[c["tire_life"] <= max_age]
    if len(c) < 4:
        return np.nan, np.nan, np.nan
    x, y = c["tire_life"].to_numpy(), c["mean_delta"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(intercept), r2