"""Scoring metrics for quantile predictions — used by every model's
validation in this repo (honesty standard, DECISIONS.md D7).

All functions take y_true plus predicted quantiles and return plain floats.
"""
from __future__ import annotations

import numpy as np


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray,
                 alpha: float) -> float:
    """Mean pinball (quantile) loss. Lower is better; alpha=0.5 -> MAE/2."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def interval_coverage(y_true: np.ndarray, p10: np.ndarray,
                      p90: np.ndarray) -> float:
    """Fraction of true values inside the P10-P90 band.

    For a calibrated model this should be ~0.80. Systematically below 0.80
    means overconfident intervals; above means too wide.
    """
    return float(np.mean((y_true >= p10) & (y_true <= p90)))


def interval_width(y_true: np.ndarray, p10: np.ndarray,
                   p90: np.ndarray) -> float:
    """Mean width of the P10-P90 band (sharpness)."""
    return float(np.mean(p90 - p10))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def quantile_scorecard(y_true, p10, p50, p90) -> dict:
    """Full honest scorecard for a P10/P50/P90 predictor."""
    y_true = np.asarray(y_true)
    return {
        "n": int(len(y_true)),
        "mae_p50_s": mae(y_true, p50),
        "rmse_p50_s": rmse(y_true, p50),
        "pinball_p10": pinball_loss(y_true, p10, 0.10),
        "pinball_p50": pinball_loss(y_true, p50, 0.50),
        "pinball_p90": pinball_loss(y_true, p90, 0.90),
        "coverage_80": interval_coverage(y_true, p10, p90),
        "mean_width_80_s": interval_width(y_true, p10, p90),
    }


def format_scorecard(sc: dict) -> str:
    lines = [
        f"n={sc['n']}",
        f"MAE(P50)={sc['mae_p50_s']:.3f}s",
        f"RMSE(P50)={sc['rmse_p50_s']:.3f}s",
        f"pinball(0.1/0.5/0.9)={sc['pinball_p10']:.3f}/{sc['pinball_p50']:.3f}/{sc['pinball_p90']:.3f}",
        f"P10-P90 coverage={sc['coverage_80']:.1%} (target ~80%)",
        f"mean width={sc['mean_width_80_s']:.2f}s",
    ]
    return " | ".join(lines)