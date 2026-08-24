"""Backtest the undercut/overcut calculator against ALL adjacent-stop pairs
in the dataset — DECISIONS.md D22 fixes 1-3.

Protocol (temporal, honest):
  For each season Y in [2022..HOLDOUT]: train the degradation model AND the
  P(attempt) model only on seasons < Y, then score EVERY adjacent-stop pair
  in season Y (not just detected attempts — scoring only attempts was the
  selection bias that made v1 anti-correlated).

Reported:
  - PRIMARY: Brier / log loss of the physics p_flip on all pairs, against
    constant-base-rate and kind-base-rate baselines.
  - Subset metrics on is_attempt==True pairs (comparable to the legacy
    detected-attempts protocol).
  - P(attempt) model quality: Brier / log loss / ROC-AUC vs its own
    constant base-rate baseline.
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
from sklearn.metrics import roc_auc_score

from pitgenius.analysis import undercut_history
from pitgenius.config import REPORTS_DIR
from pitgenius.data import store
from pitgenius.models.degradation_model import (
    TireDegradationModel,
    build_training_frame,
)
from pitgenius.strategy import undercut_calc
from pitgenius.strategy.attempt_model import AttemptModel

FIRST_TEST_YEAR = 2022

# Reference points from the pre-fix protocol (reports history, kept for
# honest before/after comparison; both scored on DETECTED ATTEMPTS ONLY):
LEGACY = {
    "v1_race_median_target": {"brier_model": 0.7141, "brier_baseline": 0.2480},
    "v2_per_stint_target": {"brier_model": 0.7590, "brier_baseline": 0.2480},
}


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(p, y, eps=1e-6):
    p = np.clip(np.asarray(p), eps, 1 - eps)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _lap_at_or_before(laps: pd.DataFrame, year, round_, driver,
                      lap) -> pd.Series | None:
    """Row for `lap`; falls back to nearest earlier lap (pit laps are
    sometimes missing from the timing feed)."""
    g = laps[(laps["year"] == year) & (laps["round"] == round_)
             & (laps["driver"] == driver) & (laps["lap_number"] <= lap)]
    if g.empty:
        return None
    return g.loc[g["lap_number"].idxmax()]


def attempt_features(laps: pd.DataFrame, att: pd.DataFrame) -> pd.DataFrame:
    """Attach tire compound/age at each car's stop lap to every pair."""
    out = []
    for _, a in att.iterrows():
        ar = _lap_at_or_before(laps, a["year"], a["round"], a["attacker"],
                               a["attacker_pit_lap"])
        dr = _lap_at_or_before(laps, a["year"], a["round"], a["defender"],
                               a["defender_pit_lap"])
        if ar is None or dr is None:
            continue
        out.append({
            **a.to_dict(),
            "attacker_compound": ar["tire_compound"],
            "attacker_age": ar["tire_life"],
            "defender_compound": dr["tire_compound"],
            "defender_age": dr["tire_life"],
            # gap convention in calc: +ve = rival leads (unchanged from the
            # legacy protocol, so before/after numbers stay comparable).
            "signed_gap_s": a["gap_at_attempt_s"]
            * (1 if a["kind"] == "undercut" else -1),
        })
    return pd.DataFrame(out)


def _block(res: pd.DataFrame) -> dict:
    base_rate = float(res["success"].mean())
    return {
        "n": int(len(res)),
        "n_success": int(res["success"].sum()),
        "base_rate": base_rate,
        "brier_model": brier(res["p_predicted"], res["success"]),
        "brier_baseline": brier(np.full(len(res), base_rate),
                                res["success"]),
        "brier_kind_baseline": brier(
            res["kind"].map(res.groupby("kind")["success"].mean()),
            res["success"]),
        "log_loss_model": log_loss(res["p_predicted"], res["success"]),
        "log_loss_baseline": log_loss(np.full(len(res), base_rate),
                                      res["success"]),
    }


def main():
    t0 = time.time()
    conn = store.connect()
    try:
        laps = store.load_laps(conn=conn)
        pits = store.load_pit_stops(conn=conn)
    finally:
        conn.close()
    print(f"Laps {len(laps):,}, stops {len(pits):,}")

    cache = REPORTS_DIR / "_pairs_cache.parquet"
    if cache.exists():
        pairs = pd.read_parquet(cache)
        print(f"Loaded {len(pairs)} cached adjacent-stop pairs")
    else:
        print("Generating ALL adjacent-stop pairs across all races ...")
        pairs = undercut_history.find_adjacent_pairs(laps, pits,
                                                     progress=True)
        pairs.to_parquet(cache, index=False)
    n_att = int(pairs["is_attempt"].sum())
    print(f"All pairs: {len(pairs)} "
          f"({pairs['kind'].value_counts().to_dict()}), "
          f"is_attempt={n_att} ({n_att / max(len(pairs), 1):.1%})")

    feats = attempt_features(laps, pairs)
    feats = feats.dropna(subset=["attacker_age", "defender_age"])
    print(f"PairsWithTireData: {len(feats)}")

    data_all = build_training_frame(laps)

    rows = []
    for year in sorted(feats["year"].unique()):
        if year < FIRST_TEST_YEAR:
            continue
        train_deg = data_all[data_all["year"] < year]
        train_att = feats[feats["year"] < year]
        test = feats[feats["year"] == year]
        if train_deg.empty or train_att.empty or test.empty:
            continue
        deg_model = TireDegradationModel().fit(train_deg)
        att_model = AttemptModel().fit(train_att)

        for _, a in test.iterrows():
            call = undercut_calc.evaluate_move(
                deg_model,
                kind=a["kind"],
                gap_s=float(a["signed_gap_s"]),
                attacker_compound=a["attacker_compound"],
                attacker_age=int(a["attacker_age"]),
                defender_compound=a["defender_compound"],
                defender_age=int(a["defender_age"]),
                circuit=f"{int(a['round'])}_{int(a['year'])}",
            )
            rows.append({
                "year": a["year"], "round": a["round"], "kind": a["kind"],
                "p_predicted": call.p_flip,
                "p_attempt": float(att_model.predict_proba(
                    pd.DataFrame([a]))[0]),
                "is_attempt": bool(a["is_attempt"]),
                "success": int(bool(a["success"])),
                "gap_at_attempt_s": a["gap_at_attempt_s"],
            })

    res = pd.DataFrame(rows)
    if res.empty:
        print("No scorable pairs found.")
        return

    all_block = _block(res)
    att_only = res[res["is_attempt"]]
    att_block = _block(att_only) if len(att_only) else {}

    # --- P(attempt) model quality (temporal holdout seasons pooled) --------
    p_att = res["p_attempt"].to_numpy()
    y_att = res["is_attempt"].astype(int).to_numpy()
    att_base = float(y_att.mean())
    attempt_report = {
        "model": "LogisticRegression(gap, ages, age_diff, same_compound)",
        "train_protocol": "seasons < each test season",
        "n_test_pairs": int(len(res)),
        "base_rate": att_base,
        "brier_model": brier(p_att, y_att),
        "brier_baseline": brier(np.full(len(res), att_base), y_att),
        "log_loss_model": log_loss(p_att, y_att),
        "log_loss_baseline": log_loss(np.full(len(res), att_base), y_att),
        "roc_auc": float(roc_auc_score(y_att, p_att)),
    }

    better = all_block["brier_model"] < all_block["brier_baseline"]
    report = {
        "protocol": (
            "Temporal: degradation model AND P(attempt) model trained only "
            "on seasons before each pair's season. ALL adjacent-stop pairs "
            "(direct neighbours, stops within 5 laps) are scored — the "
            "detected-attempts-only protocol is retained only as a labelled "
            "subset. This removes the D22 selection bias from evaluation."),
        "legacy_reference_detected_attempts_only": LEGACY,
        "primary_all_pairs": all_block,
        "subset_is_attempt": att_block,
        "attempt_model": attempt_report,
        "honest_finding": (
            "PRIMARY evaluation now scores every adjacent-stop pair, not "
            "just self-selected attempts. Result: Brier(model)"
            f"={all_block['brier_model']:.4f} vs constant baseline "
            f"{all_block['brier_baseline']:.4f} -> "
            + ("the calculator BEATS the constant baseline on all pairs."
               if better else
               "the calculator STILL DOES NOT beat the constant baseline.")
            + " P(success|attempt) and P(attempt) are modeled separately; "
              "their product is P(response AND flip), never reported as a "
              "success probability for non-attempted pairs."),
        "calibration": [],
        "by_kind": {},
        "per_pair": res.to_dict(orient="records"),
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
    print(f"\nPRIMARY (all pairs, n={all_block['n']}):")
    print(f"Brier(model)={all_block['brier_model']:.4f} vs "
          f"baseline={all_block['brier_baseline']:.4f} vs "
          f"kind-baseline={all_block['brier_kind_baseline']:.4f}")
    print(f"LogLoss(model)={all_block['log_loss_model']:.4f} vs "
          f"baseline={all_block['log_loss_baseline']:.4f}")
    if att_block:
        print(f"Subset is_attempt (n={att_block['n']}): "
              f"Brier(model)={att_block['brier_model']:.4f} vs "
              f"baseline={att_block['brier_baseline']:.4f}")
    print(f"P(attempt): Brier={attempt_report['brier_model']:.4f} vs "
          f"baseline={attempt_report['brier_baseline']:.4f}, "
          f"AUC={attempt_report['roc_auc']:.3f}")
    print(f"Wrote {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
