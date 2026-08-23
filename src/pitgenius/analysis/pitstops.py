"""Pit-stop analysis: timelines, pit-loss statistics, stop-duration distributions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pit_timeline(pits: pd.DataFrame, year: int, round_: int) -> pd.DataFrame:
    """Stops for one race sorted by lap, ready for a timeline chart."""
    p = pits[(pits["year"] == year) & (pits["round"] == round_)].copy()
    return p.sort_values(["lap", "driver"])


def pit_loss_stats(laps: pd.DataFrame, pits: pd.DataFrame,
                   max_lap_gap: int = 2) -> dict:
    """Estimate pit-lane time loss per race from out-lap vs in-lap deltas.

    For each stop we compare the driver's out-lap time (lap after the stop)
    against their own median clean lap in the same stint window. The median
    of these deltas across all stops approximates the total time lost by
    pitting (pit lane + stationary), which is the number the undercut
    calculator needs.

    Returns per-race stats plus the global distribution.
    """
    gf = laps[
        laps["lap_time_s"].notna()
        & (laps["is_pit_out_lap"] == 0)
        & (laps["track_status"].fillna("0") == "1")
    ].copy()

    med = gf.groupby(["year", "round", "driver"])["lap_time_s"].median()
    records = []
    for _, stop in pits.dropna(subset=["lap"]).iterrows():
        key = (stop["year"], stop["round"], stop["driver"])
        if key not in med.index:
            continue
        base = med.loc[key]
        # Out-lap(s): the 1-2 laps right after the stop lap.
        out = gf[
            (gf["year"] == stop["year"])
            & (gf["round"] == stop["round"])
            & (gf["driver"] == stop["driver"])
            & gf["lap_number"].between(stop["lap"] + 1, stop["lap"] + max_lap_gap)
        ]
        if out.empty:
            continue
        delta = float(out["lap_time_s"].min()) - float(base)
        records.append({
            "year": stop["year"], "round": stop["round"],
            "driver": stop["driver"], "lap": stop["lap"],
            "pit_loss_s": delta,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return {"n": 0}
    # Filter implausible deltas (SC-contaminated stops give huge negatives).
    df = df[(df["pit_loss_s"] > -5) & (df["pit_loss_s"] < 60)]
    return {
        "n": len(df),
        "median_pit_loss_s": float(df["pit_loss_s"].median()),
        "p10_pit_loss_s": float(df["pit_loss_s"].quantile(0.10)),
        "p90_pit_loss_s": float(df["pit_loss_s"].quantile(0.90)),
        "per_race": {
            (y, r): g["pit_loss_s"].median()
            for (y, r), g in df.groupby(["year", "round"])
        },
        "_rows": df,
    }


def stop_duration_distribution(pits: pd.DataFrame) -> pd.Series:
    """Distribution of official pit-stop dwell durations (seconds)."""
    d = pits["duration_s"].dropna()
    return d[(d > 15) & (d < 60)]