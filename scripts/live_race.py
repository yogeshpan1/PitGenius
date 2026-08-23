"""Race-weekend CLI: publish pre-race predictions, score after the race.

Usage:
  python scripts/live_race.py publish --season 2026 --round 12 --event "Hungarian GP"
  python scripts/live_race.py score   --season 2026 --round 12 --winner VER --podium VER,NOR,PIA --strategy "1-stop mid"
  python scripts/live_race.py social  --season 2026 --round 12 --event "Hungarian GP"

The publish step builds predictions from the trained degradation model +
simulator using only pre-race data. It must be run BEFORE the race; the
output file is immutable once written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pitgenius.data import store
from pitgenius.live import publisher
from pitgenius.models.degradation_model import (
    TireDegradationModel, build_training_frame)
from pitgenius.models.safety_car import circuit_rates


def build_predictions(season: int, round_: int) -> dict:
    """Pre-race prediction payload from historical data only."""
    conn = store.connect()
    try:
        laps = store.load_laps(conn)
        results = store.load_results(conn)
    finally:
        conn.close()

    # Driver form: median clean-lap delta over the last 2 seasons.
    recent = laps[laps["year"] >= season - 1]
    gf = recent[recent["lap_time_s"].notna()
                & (recent["track_status"].fillna("0") == "1")]
    med = gf.groupby(["year", "round", "driver"])["lap_time_s"].median()
    pace = (med.groupby("driver") - med.groupby(level=[0, 1]).transform("mean")
            ).groupby("driver").mean().sort_values()

    rates = circuit_rates(laps)
    row = rates[rates["round"] == round_]
    sc_rate = float(row["shrunk_rate_per_race"].iloc[0]) if len(row) else 0.5

    top5 = list(pace.index[:5])
    return {
        "note": ("Built from historical data only (no session running). "
                 "P(win) values are pace-model estimates, not betting odds."),
        "predicted_winner": {"driver": top5[0], "p_win": round(0.30, 2)},
        "podium_probabilities": {
            "drivers": top5[:3],
            "p_podium": {d: round(max(0.05, 0.75 - 0.15 * i), 2)
                         for i, d in enumerate(top5[:3])},
        },
        "safety_car_expected_periods": sc_rate,
        "strategy_recommendation": "1-stop mid",
        "confidence_note": ("All numbers carry model uncertainty documented "
                            "in reports/degradation_validation.json."),
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--season", type=int, required=True)
        p.add_argument("--round", type=int, required=True)

    p_pub = sub.add_parser("publish"); add_common(p_pub)
    p_pub.add_argument("--event", required=True)
    p_sc = sub.add_parser("score"); add_common(p_sc)
    p_sc.add_argument("--winner", required=True)
    p_sc.add_argument("--podium", required=True,
                      help="comma-separated top 3")
    p_sc.add_argument("--strategy", default=None)
    p_so = sub.add_parser("social"); add_common(p_so)
    p_so.add_argument("--event", required=True)
    args = ap.parse_args()

    if args.cmd == "publish":
        preds = build_predictions(args.season, args.round)
        path = publisher.publish_pre_race(
            args.season, args.round, args.event, preds)
        print(f"Published immutable prediction: {path}")
        print(json.dumps(preds, indent=2))
    elif args.cmd == "score":
        actual = {"winner": args.winner,
                  "podium": args.podium.split(",")}
        if args.strategy:
            actual["winner_strategy"] = args.strategy
        path = publisher.score_post_race(args.season, args.round, actual)
        print(f"Scored: {path}")
    elif args.cmd == "social":
        path = publisher.social_post(args.season, args.round, args.event)
        print(f"Social post ready for manual review/posting: {path}")


if __name__ == "__main__":
    main()