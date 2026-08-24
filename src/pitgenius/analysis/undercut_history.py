"""Retrospective undercut / overcut outcome analysis.

Definitions used throughout PitGenius (documented here once, reused by the
Stage 4 calculator and its backtest):

UNDERCUT attempt: driver A pits on lap L; the car directly behind A on
track (driver B) pits within L+1..L+2. The undercut "worked" if A is
ahead of B once both have completed their stops.

OVERCUT attempt: driver B (directly behind A) pits on lap L; driver A
stays out and pits within L+1..L+2. The overcut "worked" if A is still
ahead of B once both have completed their stops.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def _position_after_lap(laps: pd.DataFrame, year: int, round_: int,
                        lap: int) -> dict[str, float]:
    """Running order (by cumulative race time proxy = lap * position) at a lap.

    FastF1's per-lap Position column is noisy for cars that just pitted
    (they drop to the back of timing). We reconstruct running order from
    total elapsed time where possible; fallback is last recorded position.
    """
    sub = laps[(laps["year"] == year) & (laps["round"] == round_)]
    snap = {}
    for driver, g in sub.groupby("driver"):
        upto = g[g["lap_number"] <= lap]
        if upto.empty:
            continue
        # Cumulative time proxy: sum of clean lap times + count of missing laps
        # penalized by median lap time.
        med = g["lap_time_s"].median()
        cum = upto["lap_time_s"].fillna(med).sum()
        n_missing = lap - int(upto["lap_number"].max())
        snap[driver] = cum + n_missing * (med if pd.notna(med) else 100.0)
    return snap


def find_attempts(laps: pd.DataFrame, pits: pd.DataFrame,
                  window: int = 2) -> pd.DataFrame:
    """Detect all undercut and overcut attempts across the dataset.

    An attempt requires two cars within ~5s on track before either pits
    (proxied by being adjacent in running order at the first pit lap).
    """
    attempts = []
    races = pits.groupby(["year", "round"])
    for (year, round_), rp in races:
        lp = laps[(laps["year"] == year) & (laps["round"] == round_)]
        if lp.empty:
            continue
        stops = {
            d: sorted(g.dropna(subset=["lap"])["lap"].astype(int).tolist())
            for d, g in rp.groupby("driver")
        }
        for driver_a, stop_laps in stops.items():
            for lap_a in stop_laps:
                snap_before = _position_after_lap(lp, year, round_, lap_a - 1)
                if driver_a not in snap_before:
                    continue
                order = sorted(snap_before.items(), key=lambda kv: kv[1])
                names = [n for n, _ in order]
                if driver_a not in names:
                    continue
                idx = names.index(driver_a)
                neighbours = []
                if idx > 0:
                    neighbours.append((names[idx - 1], "ahead"))
                if idx < len(names) - 1:
                    neighbours.append((names[idx + 1], "behind"))

                for nb, rel in neighbours:
                    nb_stops = stops.get(nb, [])
                    later = [l for l in nb_stops if lap_a < l <= lap_a + window]
                    earlier = [l for l in nb_stops if lap_a - window <= l < lap_a]

                    if rel == "behind" and later:
                        # A pits first, car behind follows -> UNDERCUT attempt
                        kind = "undercut"
                        lap_b = min(later)
                    elif rel == "ahead" and earlier:
                        # Car ahead already pitted, A responds late -> OVERCUT attempt
                        kind = "overcut"
                        lap_b = max(earlier)
                    else:
                        continue

                    # Outcome: compare cumulative-time order one lap after BOTH pitted
                    settle = max(lap_a, lap_b) + 1
                    snap_after = _position_after_lap(lp, year, round_, settle)
                    if driver_a not in snap_after or nb not in snap_after:
                        continue
                    a_won = snap_after[driver_a] < snap_after[nb]
                    gap_s = abs(snap_after[nb] - snap_after[driver_a])
                    attempts.append({
                        "year": year, "round": round_,
                        "kind": kind,
                        "attacker": driver_a, "defender": nb,
                        "attacker_pit_lap": lap_a, "defender_pit_lap": lap_b,
                        "gap_at_attempt_s": abs(
                            snap_before.get(nb, np.nan)
                            - snap_before.get(driver_a, np.nan)),
                        "success": bool(a_won),
                        "settled_gap_s": gap_s,
                    })
    return pd.DataFrame(attempts)


def find_adjacent_pairs(laps: pd.DataFrame, pits: pd.DataFrame,
                        attempt_window: int = 2,
                        pair_window: int = 5,
                        progress: bool = False) -> pd.DataFrame:
    """ALL adjacent running-order pairs whose stops fall within `pair_window`
    laps of each other — not just detected attempts (DECISIONS.md D22 fixes
    2+3: removes selection bias from both modeling and evaluation).

    Canonicalisation (each physical two-car stop event appears exactly ONCE):
        attacker = the car that pitted FIRST
        defender = its direct running-order neighbour at the snapshot before
                   the first stop
        kind     = 'undercut' if the responder was the car BEHIND (it
                   followed the first pitter), else 'overcut'
        success  = attacker still ahead of defender one lap after BOTH
                   completed their stops (same outcome rule as find_attempts)
        is_attempt = the pair ALSO matches the original find_attempts
                   definition (response within `attempt_window` laps)

    The `is_attempt` column is the training label for the separate
    P(attempt) model; scoring the calculator on every pair (not just
    is_attempt==True) is the debiased evaluation.
    """
    rows = []
    races = pits.groupby(["year", "round"])
    for i_race, ((year, round_), rp) in enumerate(races):
        if progress:
            print(f"[find_adjacent_pairs] race {i_race + 1}: "
                  f"{year} r{round_} ({len(rows)} pairs so far)",
                  file=sys.stderr, flush=True)
        lp = laps[(laps["year"] == year) & (laps["round"] == round_)]
        if lp.empty:
            continue
        stops = {
            d: sorted(g.dropna(subset=["lap"])["lap"].astype(int).tolist())
            for d, g in rp.groupby("driver")
        }
        seen: set[frozenset] = set()
        for d_a, ls in stops.items():
            for lap_a in ls:
                for nb, ls_b in stops.items():
                    if nb == d_a:
                        continue
                    for lb in ls_b:
                        if lb == lap_a or abs(lb - lap_a) > pair_window:
                            continue
                        key = frozenset({(d_a, lap_a), (nb, lb)})
                        if key in seen:
                            continue

                        first_d, first_l, resp_d, resp_l = (
                            (d_a, lap_a, nb, lb) if lap_a < lb
                            else (nb, lb, d_a, lap_a))

                        snap = _position_after_lap(lp, year, round_,
                                                   first_l - 1)
                        if first_d not in snap or resp_d not in snap:
                            continue
                        order = sorted(snap.items(), key=lambda kv: kv[1])
                        names = [n for n, _ in order]
                        if (abs(names.index(first_d)
                                - names.index(resp_d)) != 1):
                            continue  # must be DIRECT neighbours, as before

                        settle = max(first_l, resp_l) + 1
                        snap_after = _position_after_lap(lp, year, round_,
                                                         settle)
                        if first_d not in snap_after or resp_d not in snap_after:
                            continue

                        resp_behind = snap[resp_d] > snap[first_d]
                        rows.append({
                            "year": year, "round": round_,
                            "kind": "undercut" if resp_behind else "overcut",
                            "attacker": first_d, "defender": resp_d,
                            "attacker_pit_lap": first_l,
                            "defender_pit_lap": resp_l,
                            "gap_at_attempt_s": abs(
                                snap[resp_d] - snap[first_d]),
                            "success": bool(
                                snap_after[first_d] < snap_after[resp_d]),
                            "settled_gap_s": abs(
                                snap_after[resp_d] - snap_after[first_d]),
                            "is_attempt": bool(
                                (resp_l - first_l) <= attempt_window),
                        })
                        seen.add(key)
    return pd.DataFrame(rows)


def summarize_attempts(attempts: pd.DataFrame) -> pd.DataFrame:
    """Success rates by kind, with counts (honesty standard D7: n shown)."""
    if attempts.empty:
        return pd.DataFrame(columns=["kind", "n", "successes", "rate"])
    g = (
        attempts.groupby("kind")
        .agg(n=("success", "size"), successes=("success", "sum"))
        .reset_index()
    )
    g["rate"] = g["successes"] / g["n"]
    return g