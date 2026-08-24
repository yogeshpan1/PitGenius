"""BoxBox cross-check (secondary dataset) — explicitly OUT of D22.

What this does:
  1. Quantifies the label sparsity in pit_strategy.csv (claimed unusable
     as a P(success|attempt) label source).
  2. Compares team_strategy.csv per-team undercut success rates (2021-22)
     against rates computed from OUR ingested data over the same seasons,
     as a rough dated sanity check only.
  3. Reports season/race coverage of the three per-lap CSVs vs the ingested
     FastF1 database, so redundancy is visible before anyone treats them as
     new signal.

It does NOT feed anything into models, the calculator, or DECISIONS.md D22.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pitgenius.analysis import undercut_history          # noqa: E402
from pitgenius.data import store                          # noqa: E402


def main() -> None:
    # ---- 1. pit_strategy.csv label sparsity --------------------------------
    ps = pd.read_csv(ROOT / "pit_strategy.csv")
    n = len(ps)
    known = ps[~ps["strategy_outcome"].fillna("Unknown").eq("Unknown")]
    print("== pit_strategy.csv ==")
    print(f"rows={n}; strategy_outcome != 'Unknown': {len(known)} "
          f"({len(known)/n:.1%}); undercut_successful non-null: "
          f"{ps['undercut_successful'].notna().sum()}")
    if len(known):
        print(known["strategy_outcome"].value_counts().to_dict())
    print("VERDICT: NOT a usable labeled dataset for P(success|attempt); "
          "excluded from D22.\n")

    # ---- 2. team_strategy.csv vs our detected attempts ---------------------
    ts = pd.read_csv(ROOT / "team_strategy.csv")
    print("== team_strategy.csv (boxbox, 2021-2022 only) ==")
    rate = (ts["successful_undercuts"] / ts["total_pit_stops"]).mean() * 100
    print(f"seasons={sorted(ts['Season'].unique())}, "
          f"teams={ts['Team'].nunique()}; unweighted mean of "
          f"'successful_undercuts/total_pit_stops': {rate:.1f}%")

    conn = store.connect()
    try:
        laps = store.load_laps(conn=conn)
        pits = store.load_pit_stops(conn=conn)
    finally:
        conn.close()
    ours = undercut_history.find_attempts(
        laps[laps["year"].isin([2021, 2022])],
        pits[pits["year"].isin([2021, 2022])])
    by_kind = ours.groupby("kind")["success"].agg(["size", "sum", "mean"])
    print("\nOur detected attempts 2021-2022 (same era, our definitions):")
    print(by_kind.to_string())
    print("NOTE: boxbox 'undercut success' counts front-car RETENTION after a "
          "response (see its avg_positions_gained semantics); our kinds use "
          "the project-wide definitions in undercut_history.py. Numbers are "
          "a sanity check on magnitude, not a label source.\n")

    # ---- 3. lap CSV coverage vs ingested DB --------------------------------
    print("== per-lap CSVs vs ingested database ==")
    db_races = laps.groupby(["year", "round"]).ngroups
    db_years = sorted(laps["year"].unique())
    print(f"DB: {db_races} races, seasons {db_years}")
    for f in ["all_race_laps_Final.csv", "2021_remaining.csv",
              "all_race_laps_2022_complete.csv"]:
        df = pd.read_csv(ROOT / f, usecols=["Season", "RaceName"])
        races = df.groupby(["Season", "RaceName"]).ngroups
        rows = len(df)
        seasons = sorted(df["Season"].unique())
        print(f"{f}: {rows:,} lap rows, {races} races, seasons {seasons}")


if __name__ == "__main__":
    main()
