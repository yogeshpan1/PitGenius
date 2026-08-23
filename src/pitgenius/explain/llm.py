"""LLM explanation layer with confidence gating (DECISIONS.md D8).

Uses any OpenAI-compatible chat API (Groq works) via env vars:
    LLM_API_KEY   (required for live mode)
    LLM_BASE_URL  (default https://api.groq.com/openai/v1)
    LLM_MODEL     (default llama-3.3-70b-versatile)

Without a key, a deterministic template stub is used and the output is
marked backend="stub". Confidence gating happens BEFORE generation: if the
underlying prediction is low-confidence, the prompt forbids strong language
and the output carries an explicit uncertainty banner.
"""
from __future__ import annotations

import json
import os
import urllib.request

from pitgenius.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

# Gating thresholds on the width of the P10-P90 band relative to |P50|.
CONFIDENT_MAX_REL_WIDTH = 0.5
UNCERTAIN_MIN_REL_WIDTH = 1.0


def gate_confidence(p10: float, p50: float, p90: float) -> str:
    """Classify a quantile prediction into a confidence tier."""
    width = p90 - p10
    rel = abs(width / p50) if p50 else float("inf")
    if rel <= CONFIDENT_MAX_REL_WIDTH:
        return "high"
    if rel >= UNCERTAIN_MIN_REL_WIDTH:
        return "low"
    return "medium"


def _stub_explanation(payload: dict, tier: str) -> str:
    pred = payload.get("prediction", {})
    lines = [
        f"[STUB - no LLM_API_KEY configured] Strategy summary:",
        f"- Call: {payload.get('call', 'n/a')}",
        f"- Median estimate: {pred.get('p50', 'n/a')}",
        f"- 80% interval: [{pred.get('p10', '?')}, {pred.get('p90', '?')}]",
        f"- Confidence tier: {tier} (band width vs median)",
    ]
    if tier != "high":
        lines.append(
            "- NOTE: interval is wide; treat this as directional only.")
    return chr(10).join(lines)


def _live_explanation(payload: dict, tier: str) -> str:
    system = (
        "You explain F1 race-strategy model outputs to racing fans. "
        f"Confidence tier: {tier}. "
        + ("Use precise, assertive language."
           if tier == "high" else
           "The model is uncertain: hedge every claim, never say "
           "'will' or 'guaranteed', prefer 'suggests'/'estimates'.")
    )
    body = json.dumps(payload)
    req = urllib.request.Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=json.dumps({
            "model": LLM_MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": body}],
            "temperature": 0.3,
        }).encode(),
        headers={"Authorization": f"Bearer {LLM_API_KEY}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def explain(payload: dict) -> dict:
    """Explain a strategy payload. Returns {text, backend, confidence}."""
    q = payload.get("prediction", {})
    tier = gate_confidence(q.get("p10", 0), q.get("p50", 1),
                           q.get("p90", 0))
    if LLM_API_KEY:
        try:
            return {"text": _live_explanation(payload, tier),
                    "backend": "live", "confidence": tier}
        except Exception as exc:  # fall back rather than fail a race weekend
            text = _stub_explanation(payload, tier)
            return {"text": text + f" [live LLM failed: {exc}]",
                    "backend": "stub_fallback", "confidence": tier}
    return {"text": _stub_explanation(payload, tier),
            "backend": "stub", "confidence": tier}