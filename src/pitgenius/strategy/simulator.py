"""Monte Carlo race simulator.

Simulates a full field (default 20 cars). Candidate strategies are attached
to separate focal cars inside the SAME simulated races, so strategy
comparisons share common random numbers (pace draws, SC schedules, pit-loss
draws) — differences between strategies are then much less noisy than
independent runs.

Physics per car per lap:
    lap_time = base_lap + pace_offset + tire_delta(compound, age)
               [+ SC slowdown, identical for everyone]
Pit stop adds pit_loss seconds (halved under SC/VSC — empirical).
tire_delta is sampled within the degradation model's P10/P50/P90 band via
uniform interpolation (same scheme as the undercut calculator).

Outputs per strategy: P(win), P(podium), expected points, and the
finishing-position distribution (P10/P50/P90). Never bare point estimates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
SC_PIT_LOSS_FACTOR = 0.45   # pit lane entry/exit still costs, less waiting
DEFAULT_FIELD_STRATEGY_STOP_FRAC = 0.42


@dataclass
class Strategy:
    name: str
    stops: list[tuple[int, str]]          # (pit lap, new compound)
    start_compound: str = "MEDIUM"


@dataclass
class SimResult:
    strategy: str
    p_win: float
    p_podium: float
    p_points: float                       # top 10
    expected_points: float
    pos_p10: float
    pos_p50: float
    pos_p90: float
    mean_race_time_s: float


POINTS_TABLE = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]


class MonteCarloSimulator:
    def __init__(
        self,
        deg_model,
        sc_rate_per_race: float,
        mean_sc_period_laps: float = 5.0,
        pit_loss_median_s: float = 20.0,
        pit_loss_sd_s: float = 1.5,
        n_field_cars: int = 19,
        field_pace_sd_s: float = 0.9,
        seed: int = 42,
    ):
        self.model = deg_model
        self.sc_rate = sc_rate_per_race
        self.mean_sc_period = mean_sc_period_laps
        self.pit_loss_med = pit_loss_median_s
        self.pit_loss_sd = pit_loss_sd_s
        self.n_field = n_field_cars
        self.field_pace_sd = field_pace_sd_s
        self.seed = seed

    # ------------------------------------------------------------- helpers
    def _quantile_tables(self, max_age: int) -> dict[str, tuple]:
        """Per compound: (q10, q50, q90) arrays indexed by tire age 0..max."""
        rows = []
        for c in COMPOUNDS:
            for age in range(1, max_age + 1):
                rows.append({
                    "tire_life": age, "fuel_proxy": 0.0,
                    "tire_compound": c, "circuit": "__sim__",
                })
        frame = pd.DataFrame(rows)
        pred = self.model.predict(frame)
        tables = {}
        k = 0
        for c in COMPOUNDS:
            q10 = np.zeros(max_age + 1)
            q50 = np.zeros(max_age + 1)
            q90 = np.zeros(max_age + 1)
            for age in range(1, max_age + 1):
                q10[age] = pred.p10[k]
                q50[age] = pred.p50[k]
                q90[age] = pred.p90[k]
                k += 1
            tables[c] = (q10, q50, q90)
        return tables

    @staticmethod
    def _sample_delta(qt: tuple[np.ndarray, np.ndarray, np.ndarray],
                      ages: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Sample deltas at given ages using uniform interpolation."""
        q10, q50, q90 = qt
        a = np.clip(ages.astype(int), 1, len(q10) - 1)
        lo = np.where(u < 0.5,
                      q10[a] + (q50[a] - q10[a]) * (u * 2),
                      q50[a] + (q90[a] - q50[a]) * ((u - 0.5) * 2))
        return lo

    # ------------------------------------------------------------------ run
    def run(self, strategies: list[Strategy], total_laps: int,
            base_lap_s: float, circuit_key: str = "__sim__",
            iterations: int = 2000) -> list[SimResult]:
        rng = np.random.default_rng(self.seed)
        max_age = min(total_laps, 45)
        qt = self._quantile_tables(max_age)

        n_strat = len(strategies)
        n_cars = n_strat + self.n_field
        results_acc = {s.name: np.zeros(iterations) for s in strategies}
        pos_acc = {s.name: np.zeros((iterations,)) for s in strategies}

        # Field default strategy: one stop around 42% distance.
        field_stop_lap = max(10, int(total_laps * DEFAULT_FIELD_STRATEGY_STOP_FRAC))

        for it in range(iterations):
            # --- safety-car schedule for this simulated race ---------------
            n_periods = min(rng.poisson(self.sc_rate), 4)
            sc_mask = np.zeros(total_laps, dtype=bool)
            for _ in range(n_periods):
                start = int(rng.integers(1, max(2, total_laps - 3)))
                length = int(max(2, rng.normal(self.mean_sc_period, 1.5)))
                sc_mask[start:min(start + length, total_laps)] = True

            # --- pace offsets ----------------------------------------------
            pace = rng.normal(0.0, self.field_pace_sd, n_cars)

            # --- state ------------------------------------------------------
            compounds = np.array(
                [s.start_compound for s in strategies]
                + ["MEDIUM"] * self.n_field
            )
            ages = np.ones(n_cars, dtype=int)
            cum = np.zeros(n_cars)
            next_stop = [list(s.stops) for s in strategies] + [
                [(field_stop_lap, "HARD")] for _ in range(self.n_field)]

            for lap in range(1, total_laps + 1):
                u = rng.uniform(size=n_cars)
                deltas = np.empty(n_cars)
                # masks must be rebuilt every lap: compounds change at stops
                for c in COMPOUNDS:
                    m = compounds == c
                    if m.any():
                        deltas[m] = self._sample_delta(qt[c], ages[m], u[m])
                lap_times = base_lap_s + pace + deltas
                if sc_mask[lap - 1]:
                    lap_times += base_lap_s * 0.28   # SC slows everyone equally
                cum += lap_times

                # pit stops this lap
                for ci in range(n_cars):
                    pending = next_stop[ci]
                    if pending and pending[0][0] == lap:
                        _, new_compound = pending.pop(0)
                        loss = rng.normal(self.pit_loss_med, self.pit_loss_sd)
                        if sc_mask[lap - 1]:
                            loss *= SC_PIT_LOSS_FACTOR
                        cum[ci] += max(loss, 12.0)
                        compounds[ci] = new_compound
                        ages[ci] = 0
                    ages[ci] += 1

            order = np.argsort(cum)          # index -> finishing position
            positions = np.empty(n_cars)
            positions[order] = np.arange(1, n_cars + 1)

            for si, s in enumerate(strategies):
                pos = positions[si]
                pos_acc[s.name][it] = pos
                pts = sum(POINTS_TABLE[:max(0, 10 - int(pos) + 1)]) \
                    if pos <= 10 else 0.0
                results_acc[s.name][it] = pts

        out = []
        for s in strategies:
            p = pos_acc[s.name]
            pts = results_acc[s.name]
            out.append(SimResult(
                strategy=s.name,
                p_win=float(np.mean(p == 1)),
                p_podium=float(np.mean(p <= 3)),
                p_points=float(np.mean(p <= 10)),
                expected_points=float(pts.mean()),
                pos_p10=float(np.percentile(p, 10)),
                pos_p50=float(np.percentile(p, 50)),
                pos_p90=float(np.percentile(p, 90)),
                mean_race_time_s=float("nan"),  # filled by caller if needed
            ))
        return out