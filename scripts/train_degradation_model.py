"""Train the tire degradation quantile model and validate it honestly.

Protocol (DECISIONS.md D13):
  A. Leave-one-race-out across all ingested races (interpolation skill).
  B. Train on seasons < HOLDOUT_YEAR, test on HOLDOUT_YEAR (deployment
     scenario — the headline number).

Outputs:
  models/degradation_quantile.joblib   final model trained on all data
  reports/degradation_validation.json  both scorecards, per-race breakdown
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from pitgenius.backtest.scoring import format_scorecard, quantile_scorecard
from pitgenius.config import REPORTS_DIR
from pitgenius.data import store
from pitgenius.models.degradation_model import (
    TireDegradationModel,
    build_training_frame,
)

HOLDOUT_YEAR = 2025


def evaluate(model: TireDegradationModel, test: pd.DataFrame) -> dict:
    pred = model.predict(test)
    return quantile_scorecard(test["delta"].to_numpy(),
                              pred.p10, pred.p50, pred.p90)


def main():
    t0 = time.time()
    conn = store.connect()
    try:
        laps = store.load_laps(conn=conn)
    finally:
        conn.close()
    print(f"Loaded {len(laps):,} lap rows "
          f"({laps.groupby(['year','round']).ngroups} races)")

    data = build_training_frame(laps)
    print(f"Training rows: {len(data):,}")
    races = sorted(data.groupby(["year", "round"]).groups.keys())
    print(f"Races: {len(races)}")

    results = {}

    # ---- A. leave-one-race-out -------------------------------------------
    loro_rows = []
    print("[A] Leave-one-race-out ...")
    for i, key in enumerate(races):
        year, round_ = key
        test = data[(data["year"] == year) & (data["round"] == round_)]
        train = data.drop(test.index)
        if len(train) == 0 or len(test) < 30:
            continue
        m = TireDegradationModel().fit(train)
        sc = evaluate(m, test)
        sc["year"], sc["round"] = year, round_
        loro_rows.append(sc)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(races)} races done ({time.time()-t0:.0f}s)")

    loro = pd.DataFrame(loro_rows)
    results["leave_one_race_out"] = {
        "aggregate": quantile_scorecard(
            np.repeat(0, 0), [], [], []) if loro.empty else {
            "n": int(loro["n"].sum()),
            "mae_p50_s": float(np.average(loro["mae_p50_s"], weights=loro["n"])),
            "rmse_p50_s": float(np.sqrt(np.average(
                loro["rmse_p50_s"] ** 2, weights=loro["n"]))),
            "pinball_p10": float(np.average(loro["pinball_p10"], weights=loro["n"])),
            "pinball_p50": float(np.average(loro["pinball_p50"], weights=loro["n"])),
            "pinball_p90": float(np.average(loro["pinball_p90"], weights=loro["n"])),
            "coverage_80": float(np.average(loro["coverage_80"], weights=loro["n"])),
            "mean_width_80_s": float(np.average(loro["mean_width_80_s"],
                                                weights=loro["n"])),
        },
        "per_race": loro.to_dict(orient="records"),
        "n_races_evaluated": int(len(loro)),
        "worst_5_races_by_mae": loro.nlargest(5, "mae_p50_s")[
            ["year", "round", "mae_p50_s", "coverage_80", "n"]
        ].to_dict(orient="records"),
    }
    print("A aggregate:", format_scorecard(results["leave_one_race_out"]["aggregate"]))

    # ---- B. season holdout -------------------------------------------------
    print(f"[B] Train <{HOLDOUT_YEAR}, test {HOLDOUT_YEAR} ...")
    train = data[data["year"] < HOLDOUT_YEAR]
    test = data[data["year"] == HOLDOUT_YEAR]
    m = TireDegradationModel().fit(train)
    holdout_sc = evaluate(m, test)
    results["season_holdout"] = {
        "holdout_year": HOLDOUT_YEAR,
        "scorecard": holdout_sc,
        "per_race": [],
    }
    per_race = []
    for key, g in test.groupby(["year", "round"]):
        p = m.predict(g)
        per_race.append({
            "year": key[0], "round": key[1],
            **quantile_scorecard(g["delta"].to_numpy(), p.p10, p.p50, p.p90),
        })
    results["season_holdout"]["per_race"] = per_race
    print("B:", format_scorecard(holdout_sc))

    # ---- Final model on all data ------------------------------------------
    print("Training final model on all data ...")
    final = TireDegradationModel().fit(data)
    path = final.save()
    results["final_model_path"] = path
    results["trained_on"] = {
        "rows": int(len(data)),
        "races": [list(map(int, k)) for k in races],
        "protocol_note": (
            "Aggregate metrics are n-weighted means of per-race scorecards. "
            "Per-race breakdowns are included unfiltered — failures included."
        ),
    }

    out = REPORTS_DIR / "degradation_validation.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {out}")
    print(f"Total time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()