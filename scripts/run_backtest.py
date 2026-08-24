"""Run the formal Stage 6 backtest across all ingested races.

Temporal protocol: for test season Y the degradation model is trained on
seasons < Y only. Writes reports/backtest_report.json and a readable
reports/backtest_report.md. Every race appears in the output.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from pitgenius.backtest.framework import backtest_race
from pitgenius.config import REPORTS_DIR
from pitgenius.data import store
from pitgenius.models.degradation_model import (
    TireDegradationModel, build_training_frame)
from pitgenius.models.pace_model import RollingPaceModel

FIRST_TEST_YEAR = 2023


def main():
    t0 = time.time()
    conn = store.connect()
    try:
        laps = store.load_laps(conn=conn)
        pits = store.load_pit_stops(conn=conn)
        results = store.load_results(conn=conn)
    finally:
        conn.close()

    data = build_training_frame(laps)
    rpm = RollingPaceModel(laps)   # built once; reused for every race
    races = sorted({(int(y), int(r)) for y, r in
                    laps.groupby(["year", "round"]).groups.keys()})

    rows = []
    current_year = None
    model = None
    for year, round_ in races:
        if year < FIRST_TEST_YEAR:
            continue
        if year != current_year:
            train = data[data["year"] < year]
            print(f"training model for {year} on {len(train)} rows ...",
                  flush=True)
            model = TireDegradationModel().fit(train)
            current_year = year
        out = backtest_race(model, laps, pits, results, year, round_,
                            rpm=rpm)
        if out:
            rows.append(out)
            mark = "HIT " if out["winner_strategy_matched"] else "MISS"
            print(f"{mark} {year} R{round_:2d} rec={out['recommended']:12s} "
                  f"winner={out['winner']}")

    df = pd.DataFrame(rows)
    n = len(df)
    hits = int(df["winner_strategy_matched"].sum()) if n else 0
    band = int(df["winner_pos_in_band"].fillna(False).sum()) if n else 0
    report = {
        "protocol": ("Simulator recommendations vs actual winner strategy; "
                     "model trained only on prior seasons. All races reported."),
        "n_races": n,
        "strategy_hits": hits,
        "strategy_misses": n - hits,
        "hit_rate": hits / n if n else None,
        "winner_pos_in_p10_p90_band": band,
        "band_rate": band / n if n else None,
        "per_race": rows,
    }
    (REPORTS_DIR / "backtest_report.json").write_text(json.dumps(report, indent=2))

    lines = [
        "# PitGenius — Formal Backtest Report",
        "",
        f"Races evaluated: **{n}** (seasons {FIRST_TEST_YEAR}+). "
        f"Model trained only on prior seasons per race.",
        "",
        f"- Simulator's recommended strategy matched the actual winner's "
        f"strategy in **{hits}/{n}** races ({hits/n:.0%}).",
        f"- Actual winner's finishing position fell inside the simulator's "
        f"P10-P90 band in **{band}/{n}** races ({band/n:.0%}).",
        "",
        "| Race | Winner | Recommended | Match | Winner P(win) | In band |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['year']} R{r['round']} | {r['winner']} | {r['recommended']} "
            f"| {'yes' if r['winner_strategy_matched'] else 'NO'} "
            f"| {r['winner_actual_p_win']:.2f} | "
            f"{'yes' if r['winner_pos_in_band'] else 'no'} |")
    (REPORTS_DIR / "backtest_report.md").write_text(chr(10).join(lines))
    print(f"wrote reports ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()