"""PitGenius — open-source F1 race strategy analytics and simulation.

Modules:
    data       — FastF1 ingestion, cleaning, SQLite/Parquet storage
    analysis   — historical tire/pit-stop/undercut analysis
    models     — tire degradation (quantile) + safety-car models
    strategy   — undercut calculator + Monte Carlo race simulator
    backtest   — formal retrospective backtesting framework
    live       — immutable pre-race prediction publishing + post-race scoring
    explain    — LLM-based, confidence-gated explanations
    simracing  — ACC / iRacing UDP telemetry bridge
    api        — FastAPI backend
"""

__version__ = "0.1.0"