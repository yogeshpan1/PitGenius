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
from sklearn.linear_model import LogisticRegression
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


def _cal_features(p_raw: np.ndarray, gap: pd.Series,
                  kind: pd.Series,
                  response_laps: pd.Series | None = None) -> np.ndarray:
    """Calibration-layer features: raw physics output, capped log-gap
    (the cumulative-time proxy has outliers up to ~200 s), kind, and the
    response window length - all known before the outcome."""
    if response_laps is None:
        response_laps = pd.Series(1, index=np.arange(len(p_raw)))
    return np.column_stack([
        np.asarray(p_raw, dtype=float),
        np.log1p(np.clip(pd.to_numeric(gap, errors="coerce").fillna(0),
                         0, 60)),
        (pd.Series(kind).values == "overcut").astype(float),
        np.clip(pd.to_numeric(response_laps, errors="coerce").fillna(1),
                1, 5).values,
    ])


# Reference points from earlier protocols (kept for honest before/after
# comparison). v1/v2 scored DETECTED ATTEMPTS ONLY; v3 introduced the
# all-pairs protocol but still had the inverted/miscalibrated p_flip.
LEGACY = {
    "v1_v2_detected_attempts_only": {
        "v1_race_median_target": {"brier_model": 0.7141,
                                  "brier_baseline": 0.2480},
        "v2_per_stint_target": {"brier_model": 0.7590,
                                "brier_baseline": 0.2480},
    },
    "v3_all_pairs_before_orientation_fix": {
        "brier_model": 0.5005, "brier_baseline": 0.1383,
        "log_loss_model": 4.3626,
    },
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
            "response_laps": int(a["defender_pit_lap"]
                                 - a["attacker_pit_lap"]),
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

        # Raw physics score for EVERY pair under this season's model. Scores
        # for prior seasons are (out-of-sample) inputs to the calibration
        # layer; nothing from season Y touches either model.
        pos_of = {idx: i for i, idx in enumerate(feats.index)}
        raw_all = np.empty(len(feats), dtype=float)
        p_att_all = np.empty(len(feats), dtype=float)
        for idx, a in feats.iterrows():
            call = undercut_calc.evaluate_move(
                deg_model,
                kind=a["kind"],
                gap_s=float(a["signed_gap_s"]),
                attacker_compound=a["attacker_compound"],
                attacker_age=int(a["attacker_age"]),
                defender_compound=a["defender_compound"],
                defender_age=int(a["defender_age"]),
                circuit=f"{int(a['round'])}_{int(a['year'])}",
                response_laps=int(a["defender_pit_lap"]
                                  - a["attacker_pit_lap"]),
            )
            raw_all[pos_of[idx]] = call.p_success
            p_att_all[pos_of[idx]] = float(att_model.predict_proba(
                pd.DataFrame([a]))[0])

        # Temporal calibration layer: logistic on prior-season pairs only.
        tr_pos = np.array([i for i, idx in enumerate(feats.index)
                           if feats.at[idx, "year"] < year])
        te_pos = np.array([pos_of[idx] for idx in test.index])
        Xtr = _cal_features(raw_all[tr_pos],
                            feats.iloc[tr_pos]["gap_at_attempt_s"],
                            feats.iloc[tr_pos]["kind"],
                            feats.iloc[tr_pos]["response_laps"])
        ytr = feats.iloc[tr_pos]["success"].astype(int).to_numpy()
        calibrator = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        p_cal_test = calibrator.predict_proba(
            _cal_features(raw_all[te_pos], test["gap_at_attempt_s"],
                          test["kind"], test["response_laps"]))[:, 1]

        for j, (_, a) in enumerate(test.iterrows()):
            rows.append({
                "year": a["year"], "round": a["round"], "kind": a["kind"],
                "p_predicted": float(p_cal_test[j]),
                "p_raw_physics": float(raw_all[te_pos[j]]),
                "p_attempt": float(p_att_all[te_pos[j]]),
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
    # Raw (uncalibrated) physics output, for transparency:
    raw_res = res.assign(p_predicted=res["p_raw_physics"])
    raw_block = _block(raw_res)

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

    report = {
        "protocol": (
            "Temporal: degradation model AND P(attempt) model trained only "
            "on seasons before each pair's season. ALL adjacent-stop pairs "
            "(direct neighbours, stops within 5 laps) are scored — the "
            "detected-attempts-only protocol is retained only as a labelled "
            "subset. This removes the D22 selection bias from evaluation."),
        "legacy_reference_detected_attempts_only": LEGACY,
        "primary_all_pairs": all_block,
        "raw_physics_before_calibration": raw_block,
        "subset_is_attempt": att_block,
        "attempt_model": attempt_report,
        "honest_finding": (
            "D22 fix 4 corrected the inverted orientation (v1-v3 returned "
            "P(first pitter LOSES) but scored against retention labels; "
            "audit showed corr=-0.328) and added multi-lap physics + a "
            "temporally-fitted calibration layer (logistic on prior-season "
            "pairs only). Result on all pairs: Brier(calibrated)="
            f"{all_block['brier_model']:.4f} vs raw physics "
            f"{raw_block['brier_model']:.4f} vs constant baseline "
            f"{all_block['brier_baseline']:.4f} and kind-baseline "
            f"{all_block['brier_kind_baseline']:.4f} -> "
            + ("the calculator now BEATS both baselines." if
               all_block["brier_model"] < min(all_block["brier_baseline"],
                                              all_block["brier_kind_baseline"])
               else "the calculator STILL does not beat the baselines.")
            + " P(attempt) remains modeled separately (AUC "
              f"{attempt_report['roc_auc']:.3f})."),
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
    print(f"Brier(calibrated)={all_block['brier_model']:.4f} vs "
          f"raw-physics={raw_block['brier_model']:.4f} vs "
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
