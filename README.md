# 🏁 PitGenius

Open-source F1 race strategy analytics and simulation platform: real timing
data in, probabilistic strategy calls out, every prediction shipped with a
validated confidence range and an honestly-graded track record.

**This is not an F1-team-grade tool.** Teams have proprietary telemetry and
years of head start. PitGenius is the most rigorously validated, honestly
self-graded strategy tool an outsider can build from public data — and it
says so everywhere it reports a number.

## What's inside

| Stage | What | Where |
|---|---|---|
| Data foundation | FastF1 ingestion → SQLite, resumable + rate-limit aware | `src/pitgenius/data/` |
| Historical analysis | Degradation curves, pit timelines, undercut/overcut outcomes, Streamlit explorer | `src/pitgenius/analysis/`, `src/pitgenius/dashboard/app.py` |
| Tire degradation model | LightGBM quantile regression (P10/P50/P90 lap-time delta) | `src/pitgenius/models/degradation_model.py` |
| Undercut calculator | Probabilistic P(flip position) on quantile samples | `src/pitgenius/strategy/undercut_calc.py` |
| Monte Carlo simulator | Full-field race sim, SC model, common random numbers | `src/pitgenius/strategy/simulator.py` |
| Backtesting | Leave-one-race-out + season holdout + full-stack retrospective | `src/pitgenius/backtest/`, `scripts/run_backtest.py` |
| Live predictions | Timestamped, hash-sealed, immutable pre-race predictions + post-race scoring | `src/pitgenius/live/publisher.py`, `scripts/live_race.py` |
| Explainability | LLM layer with confidence gating (Groq-compatible) | `src/pitgenius/explain/llm.py` |
| Tire cliff detection | Piecewise-linear breakpoint search per stint | `src/pitgenius/models/tire_cliff.py` |
| Sim racing bridge | ACC UDP parser + iRacing adapter, live pit advice | `src/pitgenius/simracing/bridge.py` |
| API + frontend | FastAPI backend, React (Vite) frontend, Streamlit demo dashboard | `src/pitgenius/api/`, `frontend/` |

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# 1. Ingest historical races (resumable; re-run to fill gaps)
python -m src.pitgenius.data.ingest --seasons 2021 2022 2023 2024 2025

# 2. Train + validate the degradation model (writes reports/)
python scripts/train_degradation_model.py

# 3. Backtests
python scripts/backtest_undercut.py
python scripts/run_backtest.py

# 4. Explore
streamlit run src/pitgenius/dashboard/app.py

# 5. Tests
pytest tests/ -q
```

## Race-weekend workflow (the live loop)

```bash
# BEFORE the race (predictions are immutable once written):
python scripts/live_race.py publish --season 2026 --round 12 --event "Hungarian GP"

# AFTER the race:
python scripts/live_race.py score --season 2026 --round 12 --winner NOR --podium NOR,PIA,VER --strategy "1-stop mid"
python scripts/live_race.py social --season 2026 --round 12 --event "Hungarian GP"
# -> predictions/<season>/r<round>/social_post.txt for manual posting
```

Every prediction file carries a UTC timestamp and SHA-256; scoring verifies
the hash first, so nothing can be quietly adjusted after the fact.

## Honesty policy

- Every model ships with its validation code **in the same stage**.
- Accuracy claims come from temporal holdouts (train on past, test on
  future), never random splits presented as skill.
- All results include counts of failures; no cherry-picking anywhere.
- See `reports/` for the actual numbers and `docs/methodology.md` for how
  they were produced.

## Credentials / stubs

| Thing | Status | What you need to do |
|---|---|---|
| LLM explainability | Stubbed behind `LLM_API_KEY` env var | Set `LLM_API_KEY` (+ optional `LLM_BASE_URL`, `LLM_MODEL`) — see `src/pitgenius/explain/llm.py` |
| AWS deployment | Config documented, not deployed | Follow `docs/deployment.md` with your AWS Academy credentials |
| Social auto-posting | Writes reviewable file only | Post manually, or add X/LinkedIn API call in `publisher.social_post` |
| ACC/iRacing live telemetry | Parsers unit-tested against synthetic packets | Run end-to-end with the sim actually running (see `docs/simracing.md`) |

## License

MIT — see LICENSE.