"""Validate the pre-race rolling pace model IN ISOLATION (D23 part 1).

Checks, with real numbers:
  1. Team-season rankings built ONLY from pre-race rolling offsets must
     reproduce known dominant/weak periods (e.g. Red Bull dominant 2023).
  2. Pre-race pace must predict same-race outcomes: rank correlation
     between pre-race offset and finishing position / grid position.
  3. Coverage: how many driver-races get an estimate at all.

No downstream model is touched here.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pitgenius.data import store
from pitgenius.models.pace_model import RollingPaceModel


def main() -> None:
    conn = store.connect()
    try:
        laps = store.load_laps(conn=conn)
        results = store.load_results(conn=conn)
    finally:
        conn.close()

    rpm = RollingPaceModel(laps)

    # ---- coverage ---------------------------------------------------------
    n_driver_races = len(results.dropna(subset=["position"]))
    have = sum(
        1 for _, r in results.dropna(subset=["position"]).iterrows()
        if rpm.driver_offset_before(int(r["year"]), int(r["round"]),
                                    r["driver"]) is not None)
    print(f"Coverage: pre-race pace available for {have:,}/{n_driver_races:,} "
          f"driver-races ({have/n_driver_races:.0%})")

    # ---- predictive correlation ------------------------------------------
    rows = []
    for _, r in results.dropna(subset=["position"]).iterrows():
        off = rpm.driver_offset_before(int(r["year"]), int(r["round"]),
                                       r["driver"])
        if off is not None:
            rows.append({"offset": off, "pos": r["position"],
                         "grid": r["grid_position"], "year": r["year"]})
    df = pd.DataFrame(rows)
    rho_pos, _ = spearmanr(df["offset"], -df["pos"])   # +rho = faster->better
    rho_grid, _ = spearmanr(df["offset"], -df["grid"])
    print(f"\nSpearman(pre-race pace, finishing position) : {rho_pos:+.3f}")
    print(f"Spearman(pre-race pace, grid position)      : {rho_grid:+.3f}")
    per_year = df.groupby("year").apply(
        lambda g: spearmanr(g["offset"], -g["pos"])[0], include_groups=False)
    print("Per-season Spearman vs finish:", 
          {int(k): round(v, 3) for k, v in per_year.items()})

    # ---- team-season rankings from PRE-RACE offsets -----------------------
    res = results.dropna(subset=["position"]).copy()
    res["pace_pre"] = [
        rpm.driver_offset_before(int(a), int(b), d) if d is not None else None
        for a, b, d in zip(res["year"], res["round"], res["driver"])]
    res = res.dropna(subset=["pace_pre"])
    ts = (res.groupby(["year", "team"])["pace_pre"]
          .agg(["mean", "size"]).reset_index())

    print("\nTeam-season rankings from PRE-RACE rolling pace "
          "(s/lap vs field; positive = faster):")
    for year, g in ts.groupby("year"):
        g = g[g["size"] >= 5].sort_values("mean", ascending=False)
        top = [f"{t} {m:+.2f}" for t, m in zip(g.head(3)["team"],
                                               g.head(3)["mean"])]
        bot = [f"{t} {m:+.2f}" for t, m in zip(g.tail(3)["team"],
                                               g.tail(3)["mean"])]
        print(f"  {year}: TOP3 {' | '.join(top)}")
        print(f"        BOT3 {' | '.join(bot)}")

    # explicit sanity anchor: Red Bull's 2023 dominance
    rb23 = ts[(ts["year"] == 2023) & (ts["team"].str.contains("Red Bull"))]
    if len(rb23):
        rank = int((ts[ts["year"] == 2023]["mean"] > rb23["mean"].iloc[0]
                    ).sum()) + 1
        print(f"\nCHECK Red Bull 2023 pre-race pace: {rb23['mean'].iloc[0]:+.3f}"
              f" s/lap -> rank {rank} of {len(ts[ts['year'] == 2023])} teams "
              f"({'PASS: rank 1' if rank == 1 else 'FAIL: not rank 1'})")


if __name__ == "__main__":
    main()
