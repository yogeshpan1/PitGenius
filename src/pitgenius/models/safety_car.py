"""Safety-car incidence model per circuit (DECISIONS.md D12).

From historical race data we count, per race, the number of distinct
interruption periods (SC and VSC) using TrackStatus transitions:
  '4' = Safety Car, '6' = VSC deployed/clearing, '7' = VSC.
A new period starts whenever status enters {4,6,7} after being out of them.

Per-circuit rate = mean periods per race, shrunk toward the global mean:
    rate_c = (n_periods_c + k * global_rate) / (n_races_c + k),  k = 3

The simulator samples period counts from a Poisson(rate_c) and period
lengths from the empirical historical distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHRINKAGE_K = 3.0
INTERRUPTION_CODES = {"4", "6", "7"}


def count_interruptions(laps: pd.DataFrame) -> pd.DataFrame:
    """Per (year, round): number of SC/VSC periods and total interruption laps."""
    rows = []
    for (year, round_), g in laps.groupby(["year", "round"]):
        g = g.sort_values("lap_number")
        status = g["track_status"].fillna("0")
        # A lap is 'interrupted' if any interruption code appears in its
        # status string (FastF1 concatenates codes, e.g. '45').
        interrupted = status.apply(lambda s: any(c in s for c in INTERRUPTION_CODES))
        # Count periods: transitions from clean -> interrupted.
        prev = interrupted.shift(1, fill_value=False)
        periods = int((interrupted & ~prev).sum())
        rows.append({
            "year": year, "round": round_,
            "n_periods": periods,
            "n_interrupted_laps": int(interrupted.sum()),
            "n_laps": int(len(g)),
        })
    return pd.DataFrame(rows)


def circuit_rates(laps: pd.DataFrame) -> pd.DataFrame:
    """Shrunk per-circuit interruption rates keyed by round number.

    Circuits are identified by round number within a season; calendars move
    year to year, so we key on round and accept that e.g. round 5 is not
    always the same track. The caller passes the current round number.
    """
    counts = count_interruptions(laps)
    if counts.empty:
        return pd.DataFrame()
    global_rate = counts["n_periods"].sum() / max(counts["n_laps"].sum() / 60, 1)

    rows = []
    for round_, g in counts.groupby("round"):
        races = len(g)
        periods = g["n_periods"].sum()
        raw_rate = periods / races if races else 0.0
        shrunk = (periods + SHRINKAGE_K * global_rate) / (races + SHRINKAGE_K)
        rows.append({
            "round": round_,
            "n_races": races,
            "raw_rate_per_race": raw_rate,
            "shrunk_rate_per_race": shrunk,
            "mean_interrupted_laps": float(g["n_interrupted_laps"].mean()),
        })
    return pd.DataFrame(rows)


def sample_interruptions(rate_per_race: float, mean_period_laps: float,
                         n_samples: int, rng: np.random.Generator,
                         max_periods: int = 4) -> list[list[int]]:
    """Sample interruption schedules: list of period lengths per sample."""
    counts = rng.poisson(rate_per_race, n_samples)
    counts = np.minimum(counts, max_periods)
    schedules = []
    for c in counts:
        schedules.append(
            [int(max(2, rng.normal(mean_period_laps, 1.5))) for _ in range(c)]
        )
    return schedules