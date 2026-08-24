"""D22 fix-1 verification probe: per-stint reference target vs race median.

Trains two degradation models on seasons < 2022 — one with the OLD
delta-vs-race-median target, one with the NEW delta-vs-per-stint-reference
target — and prints their predicted P50 deltas by tire age for the same
scenario. The diagnosed D22 pathology was implausibly fast fresh-tire deltas
(~-3 s vs driver median at tire age 2). The new target should put age-2
deltas near zero and produce a sane monotone-ish degradation slope.

Output is printed to stdout only; nothing is overwritten.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from pitgenius.analysis.degradation import green_flag_laps
from pitgenius.data import store
from pitgenius.models.degradation_model import (
    FUEL_S_PER_LAP,
    STINT_REF_MAX_AGE,
    TireDegradationModel,
    build_training_frame,
)

TRAIN_THRU_YEAR = 2021   # temporal protocol: seasons < 2022
PROBE_CIRCUIT = "10_2021"
COMPOUND = "MEDIUM"


def old_race_median_frame(laps: pd.DataFrame) -> pd.DataFrame:
    """The pre-D22-fix target: delta vs driver's whole-race median."""
    gf = green_flag_laps(laps).copy()
    med = gf.groupby(["year", "round", "driver"])["lap_time_s"].transform("median")
    gf["delta"] = gf["lap_time_s"] - med
    total = gf.groupby(["year", "round"])["lap_number"].transform("max")
    gf["fuel_proxy"] = FUEL_S_PER_LAP * (total - gf["lap_number"])
    gf["circuit"] = gf["round"].astype(str) + "_" + gf["year"].astype(str)
    return gf.dropna(subset=["delta", "tire_life"])


def _monotonicity(p50):
    """Fraction of adjacent age steps that move the wrong way (down)."""
    steps = [(b - a) for a, b in zip(p50[:-2], p50[1:-1])]
    return sum(1 for s in steps if s < -0.05) / max(len(steps), 1)


def main() -> None:
    conn = store.connect()
    try:
        laps = store.load_laps(conn=conn)
    finally:
        conn.close()

    new = build_training_frame(laps)
    print(f"NEW frame rows: {len(new):,} "
          f"(dropped, no stint ref: "
          f"{new.attrs.get('rows_dropped_no_stint_ref', 0):,})")
    # Honest breakdown of WHY rows were dropped (D22 honesty standard):
    gf_all = green_flag_laps(laps)
    n_nan_stint = int(gf_all["stint_number"].isna().sum())
    has_ref = gf_all[gf_all["tire_life"] <= STINT_REF_MAX_AGE]
    stints_with_ref = set(map(tuple,
                              has_ref[["year", "round", "driver",
                                       "stint_number"]].dropna().values))
    labelled = gf_all[gf_all["stint_number"].notna()]
    grp_cols = ["year", "round", "driver", "stint_number"]
    sizes = labelled.groupby(grp_cols, dropna=False)["lap_number"].size()
    short_stints = int((sizes < STINT_REF_MAX_AGE + 1).sum())
    print(f"  clean laps with NaN stint_number: {n_nan_stint:,}")
    print(f"  stints total: {len(sizes):,}; shorter than "
          f"{STINT_REF_MAX_AGE + 1} clean laps: {short_stints:,} "
          f"(these cannot produce a reference)")
    old = old_race_median_frame(laps)
    print(f"OLD frame rows: {len(old):,}")

    for name, df in (("OLD(race-median)", old), ("NEW(stint-ref)", new)):
        early = df[df["tire_life"] <= STINT_REF_MAX_AGE]
        print(f"{name}: mean delta at tire_life<={STINT_REF_MAX_AGE} "
              f"= {early['delta'].mean():+.3f}s over {len(early):,} rows")

    train_new = new[new["year"] <= TRAIN_THRU_YEAR]
    train_old = old[old["year"] <= TRAIN_THRU_YEAR]
    m_new = TireDegradationModel().fit(train_new)
    m_old = TireDegradationModel().fit(train_old)
    print(f"trained: OLD on {len(train_old):,} rows, "
          f"NEW on {len(train_new):,} rows (<={TRAIN_THRU_YEAR})")

    ages = list(range(2, 31, 2))
    probe = pd.DataFrame({
        "tire_life": ages,
        "fuel_proxy": [0.0] * len(ages),
        "tire_compound": [COMPOUND] * len(ages),
        "circuit": [PROBE_CIRCUIT] * len(ages),
    })
    po, pn = m_old.predict(probe), m_new.predict(probe)
    print(f"\nPredicted P50 lap delta ({COMPOUND}, circuit={PROBE_CIRCUIT}):")
    print("age   OLD(race-median)   NEW(stint-ref)")
    for i, a in enumerate(ages):
        print(f"{a:>3}   {po.p50[i]:>+8.3f} s       {pn.p50[i]:>+8.3f} s")

    # The headline checks: fresh-tire deltas must not be implausibly fast,
    # and the mid-stint curve must not FALL (the fuel-confound signature).
    fresh_old = float(po.p50[0])
    fresh_new = float(pn.p50[0])
    mid_new = float(np.mean(pn.p50[2:8]))   # ages 6..18
    late_new = float(np.mean(pn.p50[10:]))  # ages 22..30
    print(f"\nCHECK fresh-tire (age 2) delta: OLD {fresh_old:+.3f}s -> "
          f"NEW {fresh_new:+.3f}s "
          f"({'FIXED' if abs(fresh_new) < abs(fresh_old) / 2 else 'STILL SUSPECT'})")
    print(f"CHECK mid-stint (ages 6-18) mean {mid_new:+.3f}s vs late-stint "
          f"(ages 22-30) mean {late_new:+.3f}s -> "
          f"{'NO FUEL CONFOUND' if late_new >= mid_new - 0.15 else 'CURVE STILL FALLS'}")


if __name__ == "__main__":
    main()
