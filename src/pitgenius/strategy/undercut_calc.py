"""Undercut / overcut calculator, built on the Stage 3 degradation model.

Math (cumulative-time accounting, D22 fix 4 - corrected orientation):

Convention: the ATTACKER is the car that pits FIRST; gap_s is its margin
over the defender (positive = attacker LEADS by gap_s seconds, negative =
attacker trails). The defender (responder) pits response_laps laps later.
The outcome we predict matches the historical label exactly:

    success  <=>  attacker still AHEAD of defender one lap after BOTH
                  completed their stops

At that moment the attacker's lead has become:

    gap_after = gap_s
                + sum_i(old_delta_B at ages d+1 .. d+k)   # responder's k laps
                - sum_j(fresh_delta_A at ages 2 .. k+1)   # attacker's k laps
                - (P_A - P_B)

where k = response_laps. Each overlapping lap, the responder's older tires
cost it old_delta while the attacker's fresh tires cost it only fresh_delta,
so tire degradation accumulates over ALL k offset laps - the pre-fix version
modeled a single lap and ignored k entirely. P_A/P_B are pit time losses.

D22 fix 4 also corrects the ORIENTATION: v1 returned P(gap_after < 0),
i.e. the probability the first pitter LOSES, but scored it against a label
where success means the first pitter RETAINS. The orientation audit
(scripts/audit_pflip_orientation.py) showed corr(p_predicted, success) =
-0.328 on real data - predictions anti-correlated with outcomes. We now
return p_success = P(gap_after > 0).

Because fresh/old deltas come from the quantile model as P10/P50/P90, we
sample lap deltas from a triangular distribution spanning those quantiles
and report p_success plus the P10/P50/P90 of the resulting gap. This keeps
the project-wide rule: never ship a bare point estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitgenius.models.degradation_model import TireDegradationModel


@dataclass
class StrategyCall:
    kind: str                 # 'undercut' | 'overcut'
    p_success: float          # P(attacker ahead after both stops) = the
                              # same event the backtest labels `success`
    gap_after_p50: float      # median gap after both stop (+ve = attacker ahead)
    gap_after_p10: float
    gap_after_p90: float
    expected_swing_s: float   # median net swing vs no tire-age difference
    n_samples: int


def _sample_deltas(model: TireDegradationModel, rows: pd.DataFrame,
                   n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample lap-delta values consistent with the model's P10/P50/P90.

    Returns shape (n, n_rows): one sampled delta per row per draw.
    """
    pred = model.predict(rows)
    p10 = np.asarray(pred.p10)
    p50 = np.asarray(pred.p50)
    p90 = np.asarray(pred.p90)
    u = rng.uniform(size=(n, len(p50)))
    # Triangular-ish interpolation between quantiles: below-median draws
    # interpolate P10->P50, above-median draws P50->P90.
    return np.where(
        u < 0.5,
        p10 + (p50 - p10) * (u * 2),
        p50 + (p90 - p50) * ((u - 0.5) * 2),
    )


def evaluate_move(
    model: TireDegradationModel,
    *,
    kind: str,                       # 'undercut' or 'overcut' (label only;
                                     # direction is carried by sign of gap_s)
    gap_s: float,                    # attacker's margin (+ve: attacker leads)
    attacker_compound: str,
    attacker_age: int,               # attacker's tire age at its stop lap
    defender_compound: str,
    defender_age: int,               # defender's age at ITS stop lap
    circuit: str,
    response_laps: int = 1,          # laps between the two stops (k >= 1)
    fuel_proxy_attacker: float = 0.0,
    fuel_proxy_defender: float = 0.0,
    pit_loss_attacker: float = 20.0,
    pit_loss_defender: float = 20.0,
    n_samples: int = 4000,
    seed: int = 42,
) -> StrategyCall:
    """P(attacker ahead after both cars complete their stops)."""
    if kind not in ("undercut", "overcut"):
        raise ValueError("kind must be 'undercut' or 'overcut'")
    k = max(int(response_laps), 1)
    rng = np.random.default_rng(seed)

    rows = pd.DataFrame({
        # attacker's k full laps after its stop: fresh tires, ages 2..k+1
        "tire_life": list(range(2, k + 2))
        # defender's k laps before/at its stop: ages d+1 .. d+k
        + [max(defender_age + 1, 2) + i for i in range(k)],
        "fuel_proxy": [fuel_proxy_attacker] * k + [fuel_proxy_defender] * k,
        "tire_compound": [attacker_compound] * k + [defender_compound] * k,
        "circuit": [circuit] * (2 * k),
    })
    samples = _sample_deltas(model, rows, n_samples, rng)
    fresh_sum = samples[:, :k].sum(axis=1)   # attacker's k fresh laps
    old_sum = samples[:, k:].sum(axis=1)     # defender's k old-tire laps

    # Car-to-car pit-loss difference uncertainty (~0.5s sd; same-team stops
    # differ by fractions of a second, cross-team up to ~1s).
    p_a = rng.normal(pit_loss_attacker, 0.5, n_samples)
    p_b = rng.normal(pit_loss_defender, 0.5, n_samples)

    # gap_after from ATTACKER's perspective: positive => attacker ahead.
    gap_after = gap_s + (old_sum - fresh_sum) - (p_a - p_b)
    p_success = float(np.mean(gap_after > 0))

    return StrategyCall(
        kind=kind,
        p_success=p_success,
        gap_after_p50=float(np.percentile(gap_after, 50)),
        gap_after_p10=float(np.percentile(gap_after, 10)),
        gap_after_p90=float(np.percentile(gap_after, 90)),
        expected_swing_s=float(np.median((old_sum - fresh_sum)
                                         - (p_a - p_b))),
        n_samples=n_samples,
    )


def recommend(
    model: TireDegradationModel,
    gap_s: float,
    compounds: tuple[str, str],
    ages: tuple[int, int],
    circuit: str,
    response_laps: int = 2,
    **kwargs,
) -> dict:
    """Convenience wrapper: should the TRAILING car pit NOW?

    gap_s positive = rival leads. The trailing car boxing now becomes the
    first pitter and currently TRAILS, so it enters the physics with a
    negative margin; p_success is its probability of ending ahead.
    """
    call = evaluate_move(
        model, kind="undercut", gap_s=-abs(float(gap_s)),
        attacker_compound=compounds[0], attacker_age=ages[0],
        defender_compound=compounds[1], defender_age=ages[1],
        circuit=circuit, response_laps=response_laps, **kwargs,
    )
    verdict = ("UNDERCUT ON" if call.p_success >= 0.6 else
               "MARGINAL" if call.p_success >= 0.45 else "STAY OUT")
    return {
        "verdict": verdict,
        "p_gain_position": round(call.p_success, 3),
        "gap_after_p10_p50_p90": (
            round(call.gap_after_p10, 1),
            round(call.gap_after_p50, 1),
            round(call.gap_after_p90, 1),
        ),
        "detail": call,
    }
