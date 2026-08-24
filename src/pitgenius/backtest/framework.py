"""Formal retrospective backtest of the full strategy stack.

For every historical race, using ONLY information available before that
season (temporal protocol, D13):
  1. Build candidate strategies (1-stop early/mid/late).
  2. Run the Monte Carlo simulator with the era-appropriate degradation
     model and the circuit's historical SC rate.
  3. Compare the simulator's recommended strategy against the strategy the
     actual race winner used.
  4. Check whether the actual winner's finishing position falls inside the
     simulator's P10-P90 band for that strategy.

Honesty: every race is reported, including races where the recommendation
was wrong. Aggregates are never shown without counts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pitgenius.models.pace_model import RollingPaceModel
from pitgenius.models.safety_car import circuit_rates
from pitgenius.strategy.simulator import MonteCarloSimulator, Strategy


def winner_strategy(laps: pd.DataFrame, pits: pd.DataFrame,
                    results: pd.DataFrame, year: int, round_: int):
    """Reconstruct the race winner's stop laps + compounds."""
    res = results[(results["year"] == year) & (results["round"] == round_)]
    fin = res.dropna(subset=["position"])
    if fin.empty:
        return None, None
    winner = fin.loc[fin["position"].idxmin(), "driver"]
    wp = pits[(pits["year"] == year) & (pits["round"] == round_)
              & (pits["driver"] == winner)].sort_values("lap")
    wl = laps[(laps["year"] == year) & (laps["round"] == round_)
              & (laps["driver"] == winner)].sort_values("lap_number")
    if wl.empty:
        return winner, None
    stops = []
    for _, s in wp.iterrows():
        row = wl[wl["lap_number"] == s["lap"] + 1]
        comp = row["tire_compound"].iloc[0] if len(row) else "MEDIUM"
        stops.append((int(s["lap"]), comp))
    start = wl["tire_compound"].iloc[0]
    return winner, Strategy(name="winner_actual", stops=stops,
                            start_compound=start)


def candidate_strategies(total_laps: int) -> list[Strategy]:
    """Three canonical one-stop plans: early / mid / late."""
    mid = int(total_laps * 0.45)
    return [
        Strategy("1-stop early", [(int(total_laps * 0.30), "MEDIUM")]),
        Strategy("1-stop mid", [(mid, "HARD")]),
        Strategy("1-stop late", [(int(total_laps * 0.60), "HARD")]),
    ]


def backtest_race(model, laps, pits, results, year, round_,
                  iterations: int = 300,
                  rpm: RollingPaceModel | None = None) -> dict | None:
    race_laps = laps[(laps["year"] == year) & (laps["round"] == round_)]
    if race_laps.empty:
        return None
    total_laps = int(race_laps["lap_number"].max())
    if total_laps < 30:
        return None
    gf = race_laps[race_laps["lap_time_s"].notna()
                   & (race_laps["track_status"].fillna("0") == "1")]
    if gf.empty:
        return None
    base_lap = float(gf["lap_time_s"].median())

    rates = circuit_rates(laps)
    row = rates[rates["round"] == round_]
    sc_rate = float(row["shrunk_rate_per_race"].iloc[0]) if len(row) else 0.5
    mean_period = float(row["mean_interrupted_laps"].iloc[0]) if len(row) else 5.0

    winner, w_strategy = winner_strategy(laps, pits, results, year, round_)
    if w_strategy is None:
        return None

    cands = candidate_strategies(total_laps)
    if w_strategy.stops:
        cands.append(w_strategy)

    # D23 part 3: real PRE-RACE pace offsets per car (no leakage -
    # RollingPaceModel uses only strictly-prior races). Candidate-strategy
    # focal cars get the field-median offset so STRATEGY is what differs
    # between them; the winner_actual focal car gets the actual winner's
    # offset; field cars get their own offsets (median fallback for drivers
    # without history). SIGN: pace_model offsets are positive=faster, the
    # simulator ADDS its pace term to lap time, so negate on the way in.
    if rpm is None:
        rpm = RollingPaceModel(laps)
    offs = rpm.offsets_before(year, round_)
    known = np.array(list(offs.values())) if offs else np.array([0.0])
    med_pace = float(np.median(known))
    winner_pace = float(offs.get(winner, med_pace))
    fin_all = results[(results["year"] == year)
                      & (results["round"] == round_)].dropna(subset=["position"])
    others = [d for d in fin_all.sort_values("position")["driver"]
              if d != winner][:19]
    field_pace = [-float(offs.get(d, med_pace)) for d in others]
    # cands[-1] is always winner_actual (appended above), so the LAST focal
    # slot carries the winner's pace term; other focals stay field-median.
    focal_pace = [-med_pace] * (len(cands) - 1) + [-winner_pace]
    pace_offsets = focal_pace + field_pace + [-med_pace] * 19

    sim = MonteCarloSimulator(model, sc_rate, mean_period,
                              pace_offsets=pace_offsets)
    sims = sim.run(cands, total_laps, base_lap, iterations=iterations)

    best = max(sims, key=lambda r: r.expected_points)
    actual = next((r for r in sims if r.strategy == "winner_actual"), None)
    res_pos = results[(results["year"] == year) & (results["round"] == round_)]
    fin = res_pos.dropna(subset=["position"])
    winner_pos = int(fin.loc[fin["driver"] == winner, "position"].iloc[0]) \
        if winner and (fin["driver"] == winner).any() else None

    return {
        "year": year, "round": round_, "winner": winner,
        "total_laps": total_laps, "sc_rate": sc_rate,
        "recommended": best.strategy,
        "recommended_p_win": best.p_win,
        "winner_strategy_matched": best.strategy == "winner_actual",
        "winner_actual_p_win": actual.p_win if actual else None,
        "winner_pos_in_band": (
            actual.pos_p10 <= winner_pos <= actual.pos_p90
            if actual and winner_pos else None),
        "winner_pos": winner_pos,
        "per_strategy": [
            {"strategy": r.strategy, "p_win": r.p_win,
             "p_podium": r.p_podium, "pos_p50": r.pos_p50}
            for r in sims
        ],
    }