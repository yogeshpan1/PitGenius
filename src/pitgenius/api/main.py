"""PitGenius FastAPI backend.

Run: uvicorn src.pitgenius.api.main:app --reload
Endpoints serve the SQLite store, the trained degradation model, and the
immutable prediction files. The React frontend (frontend/) consumes these.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pitgenius.data import store
from pitgenius.live import publisher

app = FastAPI(title="PitGenius API", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/races")
def races():
    df = store.load_races()
    return df.to_dict(orient="records")


@app.get("/api/races/{year}/{round}/laps")
def race_laps(year: int, round: int):
    df = store.load_laps(year=year, round_=round)
    if df.empty:
        raise HTTPException(404, "race not ingested")
    return df.to_dict(orient="records")


@app.get("/api/races/{year}/{round}/results")
def race_results(year: int, round: int):
    df = store.load_results(year=year, round_=round)
    return df.to_dict(orient="records")


@app.get("/api/predictions/{season}/{round}")
def get_prediction(season: int, round: int):
    try:
        return publisher.load_published(season, round)
    except FileNotFoundError:
        raise HTTPException(404, "no published prediction for this race")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/health")
def health():
    n = len(store.load_races())
    return {"status": "ok", "races_ingested": n}