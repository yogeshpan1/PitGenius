"""Tire cliff detection: find the lap where degradation suddenly spikes.

Method: fit piecewise-linear (one breakpoint) to a stint's lap-time deltas
vs tire age; the breakpoint is the cliff lap. Search over candidate
breakpoints, pick the one minimizing SSE, require the post-cliff slope to
exceed the pre-cliff slope by a margin, else declare 'no cliff'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_MARGIN_S_PER_LAP = 0.05   # post slope must exceed pre slope by this


def _fit_two_segment(x: np.ndarray, y: np.ndarray, bp: int):
    left = x <= bp
    if left.sum() < 3 or (~left).sum() < 3:
        return None
    p1 = np.polyfit(x[left], y[left], 1)
    p2 = np.polyfit(x[~left], y[~left], 1)
    pred = np.where(left, np.polyval(p1, x), np.polyval(p2, x))
    sse = float(np.sum((y - pred) ** 2))
    return sse, p1[0], p2[0]


def detect_cliff(stint_laps: pd.DataFrame, delta_col: str = "delta",
                 age_col: str = "tire_life") -> dict:
    """stint_laps: one driver's single-stint clean laps."""
    df = stint_laps.dropna(subset=[delta_col]).sort_values(age_col)
    x = df[age_col].to_numpy(dtype=float)
    y = df[delta_col].to_numpy(dtype=float)
    if len(df) < 8:
        return {"cliff_lap": None, "reason": "too few laps"}

    best = None
    for bp in range(int(x.min()) + 3, int(x.max()) - 2):
        fit = _fit_two_segment(x, y, bp)
        if fit and (best is None or fit[0] < best[1][0]):
            best = (bp, fit)

    if best is None:
        return {"cliff_lap": None, "reason": "no valid breakpoint"}
    bp, (sse, pre, post) = best
    if post - pre < MIN_MARGIN_S_PER_LAP:
        return {"cliff_lap": None, "reason": "no significant cliff",
                "pre_slope": pre, "post_slope": post}
    return {"cliff_lap": int(bp), "pre_slope_s_per_lap": round(pre, 4),
            "post_slope_s_per_lap": round(post, 4), "sse": round(sse, 3)}


def detect_race_cliffs(laps: pd.DataFrame) -> pd.DataFrame:
    """Detect cliffs for every driver stint in a race's laps DataFrame."""
    gf = laps[laps["lap_time_s"].notna()
              & (laps["is_pit_out_lap"] == 0)
              & (laps["track_status"].fillna("0") == "1")].copy()
    med = gf.groupby("driver")["lap_time_s"].transform("median")
    gf["delta"] = gf["lap_time_s"] - med
    rows = []
    for (driver, stint), g in gf.groupby(["driver", "stint_number"]):
        r = detect_cliff(g)
        rows.append({"driver": driver, "stint": stint,
                     "compound": g["tire_compound"].iloc[0],
                     "n_laps": len(g), **r})
    return pd.DataFrame(rows)