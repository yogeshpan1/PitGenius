"""Central configuration: paths, constants, environment-driven settings."""
from __future__ import annotations

import os
from pathlib import Path

# Repo root = two levels above this file (src/pitgenius/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"          # FastF1 HTTP cache
RAW_DIR = DATA_DIR / "raw"              # cleaned parquet dumps
DB_PATH = DATA_DIR / "pitgenius.db"     # SQLite store
MODELS_DIR = PROJECT_ROOT / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _d in (DATA_DIR, CACHE_DIR, RAW_DIR, MODELS_DIR, PREDICTIONS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Ingestion -----------------------------------------------------------
SEASONS_2021_2025 = [2021, 2022, 2023, 2024, 2025]
RACE_SESSION_CODE = "R"

# --- LLM explainability (Stage 8) ----------------------------------------
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

# --- Monte Carlo defaults -------------------------------------------------
MC_DEFAULT_ITERATIONS = 5000
MC_RANDOM_SEED = 42

# Quantiles used for every prediction surface in the project.
PREDICTION_QUANTILES = (0.10, 0.50, 0.90)


def format_lap_time(seconds: float | None) -> str:
    """Format seconds as m:ss.mmm for display. None/NaN -> '—'."""
    import math

    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "—"
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes}:{secs:06.3f}"