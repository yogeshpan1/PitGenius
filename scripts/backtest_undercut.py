"""Backtest the undercut/overcut calculator against every real historical
attempt in the dataset.

Protocol (temporal, honest):
  For each season Y in [2022..HOLDOUT]: train the degradation model only on
  seasons < Y, then score every detected attempt in season Y.
  Reported: Brier score, log loss, calibration by predicted-probability
  bucket, counts of successes AND failures, plus a base-rate baseline.

Outputs: reports/undercut_backtest.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from pitgenius.analysis import undercut_history
from pitgenius.config import REPORTS_DIR
from pitgenius.data import store
from pitgenius.models.degradation_model import (
    TireDegradationModel,
    build_training_frame,
)
from pitgenius.strategy import undercut_calc

FIRST_TEST_YEAR = 2022


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(p, y, eps=1e-6):
    p = np.clip(np.asarray(p), eps, 1 - eps)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def attempt_features(laps: pd.DataFrame, att: pd.DataFrame) -> pd.DataFrame:
    """Attach tire compound/age at each car's stop lap to every attempt."""
    out = []
    lap_idx = laps.set_index(["year", "round", "driver", "lap_number"])
    for _, a in att.iterrows():
        try:
            ar = lap_idx.loc[(a["year"], a["round"], a["attacker"],
                              a["attacker_pit_lap"])]
            dr = lap_idx.loc[(a["year"], a["round"], a["defender"],
                              a["defender_pit_lap"])]
        except KeyError:
            continue
        if isinstance(ar, pd.DataFrame):
            ar = ar.iloc[0]
        if isinstance(dr, pd.DataFrame):
            dr = dr.iloc[0]
        out.append({
            **a.to_dict(),
            "attacker_compound": ar["tire_compound"],
            "attacker_age": ar["tire_life"],
            "defender_compound": dr["tire_compound"],
            "defender_age": dr["tire_life"],
            # gap convention in calc: +ve = rival leads. In `find_attempts`,
            # attacker is behind for undercuts (gap>0), ahead for overcuts.
            "signed_gap_s": a["gap_at_attempt_s"]
            * (1 if a["kind"] == "undercut" else -1),
        })
    return pd.DataFrame(out)


def main():
    t0 = time.time()
    conn = store.connect()
    try:
        laps = store.load_laps(conn)
        pits = store.load_pit_stops(conn)
    finally:
        conn.close()
    print(f"Laps {len(laps):,}, stops {len(pits):,}")

    cache = REPORTS_DIR / "_attempts_cache.parquet"
    if cache.exists():
        att = pd.read_parquet(cache)
        print(f"Loaded {len(att)} cached attempts")
    else:
        print("Detecting attempts across all races ...")
        att = undercut_history.find_attempts(laps, pits)
        att.to_parquet(cache, index=False)
    print(f"Attempts detected: {len(att)} "
          f"({att['kind'].value_counts().to_dict() if len(att) else {}})")

    feats = attempt_features(laps, att)
    print(f"AttemptsWithTireData: {len(feats)}")

    data_all = build_training_frame(laps)

    rows = []
    for year in sorted(feats["year"].unique()):
        if year < FIRST_TEST_YEAR:
            continue
        train = data_all[data_all["year"] < year]
        test = feats[feats["year"] == year]
        if train.empty or test.empty:
            continue
        model = TireDegradationModel().fit(train)

        preds, ys = [], []
        for _, a in test.iterrows():
            call = undercut_calc.evaluate_move(
                model,
                kind=a["kind"],
                gap_s=float(a["signed_gap_s"]),
                attacker_compound=a["attacker_compound"],
                attacker_age=int(a["attacker_age"]),
                defender_compound=a["defender_compound"],
                defender_age=int(a["defender_age"]),
                circuit=f"{int(a['round'])}_{int(a['year'])}",
            )
            preds.append(call.p_flip)
            ys.append(int(bool(a["success"])))
            rows.append({
                "year": a["year"], "round": a["round"], "kind": a["kind"],
                "p_predicted": call.p_flip, "success": int(bool(a["success"])),
                "gap_at_attempt_s": a["gap_at_attempt_s"],
            })

    res = pd.DataFrame(rows)
    if res.empty:
        print("No scorable attempts found.")
        return

    base_rate = float(res["success"].mean())
    report = {
        "protocol": ("Temporal: degradation model trained only on seasons "
                     "before each attempt's season. All detected attempts "
                     "scored — no filtering."),
        "n_attempts": int(len(res)),
        "n_success": int(res["success"].sum()),
        "n_failure": int(len(res) - res["success"].sum()),
        "base_rate": base_rate,
        "brier_model": brier(res["p_predicted"], res["success"]),
        "brier_baseline": brier(np.full(len(res), base_rate),
                                res["success"]),
        "log_loss_model": log_loss(res["p_predicted"], res["success"]),
        "log_loss_baseline": log_loss(np.full(len(res), base_rate),
                                      res["success"]),
        "calibration": [],
        "by_kind": {},
        "per_attempt": res.to_dict(orient="records"),
    }

    bins = [(0.0, 0.3), (0.3, 0.45), (0.45, 0.55), (0.55, 0.7), (0.7, 1.01)]
    for lo, hi in bins:
        m = (res["p_predicted"] >= lo) & (res["p_predicted"] < hi)
        if m.sum():
            report["calibration"].append({
                "bucket": f"[{lo:.2f},{hi:.2f})",
                "n": int(m.sum()),
                "mean_predicted": float(res.loc[m, "p_predicted"].mean()),
                "actual_rate": float(res.loc[m, "success"].mean()),
            })
    for kind, g in res.groupby("kind"):
        report["by_kind"][kind] = {
            "n": int(len(g)), "successes": int(g["success"].sum()),
            "base_rate": float(g["success"].mean()),
            "brier_model": brier(g["p_predicted"], g["success"]),
        }

    out = REPORTS_DIR / "undercut_backtest.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Brier(model)={report['brier_model']:.4f} vs "
          f"baseline={report['brier_baseline']:.4f}")
    print(f"LogLoss(model)={report['log_loss_model']:.4f} vs "
          f"baseline={report['log_loss_baseline']:.4f}")
    print(f"Wrote {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()