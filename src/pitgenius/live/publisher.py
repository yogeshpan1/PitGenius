"""Live prediction infrastructure: pre-race publishing, post-race scoring.

Immutability contract (DECISIONS.md D9):
  - predictions/<season>/r<round>/pre_race.json is written ONCE with a UTC
    timestamp and SHA-256 of its content. Publishing again raises unless
    force=True (which archives the old file first).
  - Scoring writes post_race_score.json alongside; it never edits the
    prediction file and verifies its hash before scoring.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from pitgenius.config import PREDICTIONS_DIR


def _race_dir(season: int, round_: int) -> Path:
    d = PREDICTIONS_DIR / str(season) / f"r{round_}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(obj: dict) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode()).hexdigest()


def publish_pre_race(season: int, round_: int, event_name: str,
                     predictions: dict, force: bool = False) -> Path:
    """Write the immutable pre-race prediction file."""
    d = _race_dir(season, round_)
    out = d / "pre_race.json"
    if out.exists() and not force:
        raise FileExistsError(
            f"{out} already published; predictions are immutable. "
            f"Use force=True to archive and republish.")
    if out.exists() and force:
        archived = d / f"pre_race.archived.{out.stat().st_mtime_ns}.json"
        archived.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")

    payload = {
        "schema": "pitgenius.pre_race.v1",
        "season": season, "round": round_, "event_name": event_name,
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictions": predictions,
    }
    payload["content_sha256"] = _sha256(payload)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    return out


def load_published(season: int, round_: int) -> dict:
    out = _race_dir(season, round_) / "pre_race.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    expected = payload.pop("content_sha256")
    if _sha256(payload) != expected:
        raise ValueError(f"hash mismatch on {out}: modified after publication")
    payload["content_sha256"] = expected
    return payload


def score_post_race(season: int, round_: int, actual: dict) -> Path:
    """Grade the immutable prediction against actual outcome."""
    pred = load_published(season, round_)   # raises on tampering
    p = pred["predictions"]
    checks = {}
    if "predicted_winner" in p:
        checks["winner_correct"] = (
            p["predicted_winner"].get("driver") == actual.get("winner"))
    if "podium_probabilities" in p and "podium" in actual:
        predicted_top3 = set(p["podium_probabilities"].get("drivers", []))
        checks["podium_overlap"] = len(predicted_top3 & set(actual["podium"]))
    if "strategy_recommendation" in p and "winner_strategy" in actual:
        checks["strategy_matched_winner"] = (
            p["strategy_recommendation"] == actual["winner_strategy"])

    score = {
        "schema": "pitgenius.post_race.v1",
        "season": season, "round": round_,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_published_at_utc": pred["published_at_utc"],
        "prediction_sha256_verified": True,
        "checks": checks, "actual": actual,
    }
    out = _race_dir(season, round_) / "post_race_score.json"
    out.write_text(json.dumps(score, indent=2), encoding="utf-8")
    return out


def social_post(season: int, round_: int, event_name: str) -> Path:
    """Generate reviewable social content from a scored race.

    No auto-posting: X/LinkedIn credentials are intentionally not assumed.
    Automating later only needs an API call appended here.
    """
    d = _race_dir(season, round_)
    score_path = d / "post_race_score.json"
    if not score_path.exists():
        raise FileNotFoundError("score the race first (score_post_race)")
    score = json.loads(score_path.read_text(encoding="utf-8"))
    pred = json.loads((d / "pre_race.json").read_text(encoding="utf-8"))

    checks = score["checks"]
    lines = [
        f"PitGenius pre-race prediction vs reality — {event_name} "
        f"({season} R{round_})", "",
        f"Published before the race at {pred['published_at_utc']} "
        f"(immutable, hash-verified).", "",
    ]
    if "winner_correct" in checks:
        w = checks["winner_correct"]
        pw = pred["predictions"]["predicted_winner"]
        lines.append(
            f"Winner call: {'CORRECT' if w else 'WRONG'} — predicted "
            f"{pw.get('driver')} (P(win)={pw.get('p_win', 0):.0%}), "
            f"actual {score['actual'].get('winner')}.")
    if "podium_overlap" in checks:
        lines.append(f"Podium overlap: {checks['podium_overlap']}/3.")
    if "strategy_matched_winner" in checks:
        lines.append(
            f"Strategy call matched winner's actual strategy: "
            f"{'yes' if checks['strategy_matched_winner'] else 'no'}.")
    lines += ["", "#F1 #RaceStrategy #PitGenius"]
    out = d / "social_post.txt"
    out.write_text(chr(10).join(lines), encoding="utf-8")
    return out