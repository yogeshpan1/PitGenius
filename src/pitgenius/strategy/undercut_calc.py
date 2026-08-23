"""Undercut / overcut calculator, built on the Stage 3 degradation model.

Math (cumulative-time accounting):

Attacker A is BEHIND defender B by gap g seconds (g > 0 means B leads).
A pits on lap L, B pits on lap L+1. At the moment B completes its stop:

    gap_after = g + fresh_delta_A - old_delta_B - (P_A - P_B)

where
    fresh_delta_A ~ predicted delta of A's next lap on fresh tires
    old_delta_B   ~ predicted delta of B's next lap on tires one lap older
    P_x           ~ pit time loss for car x (assumed equal unless given)

The undercut FLIPS the positions iff gap_after < 0.

Because fresh/old deltas come from the quantile model as P10/P50/P90, we
sample lap deltas from a triangular distribution spanning those quantiles
and report P(flip) plus the P10/P50/P90 of the resulting gap. This keeps the
project-wide rule: never ship a bare point estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitgenius.models.degradation_model import TireDegradationModel


@dataclass
class StrategyCall:
    kind: str                 # 'undercut' | 'overcut'
    p_flip: float             # probability the move gains the position
    gap_after_p50: float      # median gap after both stop (+ve = still behind)
    gap_after_p10: float
    gap_after_p90: float
    expected_swing_s: float   # median time gained vs staying out together
    n_samples: int


def _sample_deltas(model: TireDegradationModel, rows: pd.DataFrame,
                   n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample lap-delta values consistent with the model's P10/P50/P90."""
    pred = model.predict(rows)
    u = rng.uniform(size=n)
    # Triangular-ish interpolation between quantiles: below-median draws
    # interpolate P10->P50, above-median draws P50->P90.
    return np.where(
        u < 0.5,
        pred.p10 + (pred.p50 - pred.p10) * (u * 2),
        pred.p50 + (pred.p90 - pred.p50) * ((u - 0.5) * 2),
    )


def evaluate_move(
    model: TireDegradationModel,
    *,
    kind: str,                       # 'undercut' or 'overcut'
    gap_s: float,                    # +ve: rival leads by this much
    attacker_compound: str,
    attacker_age: int,               # attacker's tire age at its stop lap
    defender_compound: str,
    defender_age: int,               # defender's age one lap later
    circuit: str,
    fuel_proxy_attacker: float = 0.0,
    fuel_proxy_defender: float = 0.0,
    pit_loss_attacker: float = 20.0,
    pit_loss_defender: float = 20.0,
    n_samples: int = 4000,
    seed: int = 42,
) -> StrategyCall:
    """Probability that the early/late stop gains the position."""
    if kind not in ("undercut", "overcut"):
        raise ValueError("kind must be 'undercut' or 'overcut'")
    rng = np.random.default_rng(seed)

    rows = pd.DataFrame({
        "tire_life": [2, max(defender_age + 1, 3)],
        "fuel_proxy": [fuel_proxy_attacker, fuel_proxy_defender],
        "tire_compound": [attacker_compound, defender_compound],
        "circuit": [circuit, circuit],
    })
    samples = _sample_deltas(model, rows, n_samples, rng)
    fresh = samples[:, 0]     # attacker out-lap on ~2-lap-old tires
    old = samples[:, 1]       # defender's lap on older tires

    # Pit-loss uncertainty: +/-1.5s empirical spread (see reports).
    p_a = rng.normal(pit_loss_attacker, 1.5, n_samples)
    p_b = rng.normal(pit_loss_defender, 1.5, n_samples)

    # gap_after from ATTACKER's perspective: negative => ahead.
    gap_after = gap_s + fresh - old - (p_a - p_b)
    p_flip = float(np.mean(gap_after < 0))

    return StrategyCall(
        kind=kind,
        p_flip=p_flip,
        gap_after_p50=float(np.percentile(gap_after, 50)),
        gap_after_p10=float(np.percentile(gap_after, 10)),
        gap_after_p90=float(np.percentile(gap_after, 90)),
        expected_swing_s=float(np.median((p_b - p_a) + old - fresh)),
        n_samples=n_samples,
    )


def recommend(
    model: TireDegradationModel,
    gap_s: float,
    compounds: tuple[str, str],
    ages: tuple[int, int],
    circuit: str,
    **kwargs,
) -> dict:
    """Convenience wrapper: should the trailing car pit NOW (undercut)?"""
    call = evaluate_move(
        model, kind="undercut", gap_s=gap_s,
        attacker_compound=compounds[0], attacker_age=ages[0],
        defender_compound=compounds[1], defender_age=ages[1],
        circuit=circuit, **kwargs,
    )
    verdict = ("UNDERCUT ON" if call.p_flip >= 0.6 else
               "MARGINAL" if call.p_flip >= 0.45 else "STAY OUT")
    return {
        "verdict": verdict,
        "p_gain_position": round(call.p_flip, 3),
        "gap_after_p10_p50_p90": (
            round(call.gap_after_p10, 1),
            round(call.gap_after_p50, 1),
            round(call.gap_after_p90, 1),
        ),
        "detail": call,
    }