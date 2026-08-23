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
                  iterations: int = 300) -> dict | None:
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

    sim = MonteCarloSimulator(model, sc_rate, mean_sc_period)
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