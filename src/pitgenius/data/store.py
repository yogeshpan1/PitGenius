"""SQLite storage layer for cleaned race data.

Schema (see DECISIONS.md D1/D2 — durations are float seconds):
    races               one row per (year, round) event
    laps                one row per driver per lap
    pit_stops           one row per pit stop
    results             final classification per race
    ingestion_manifest  idempotency ledger for the ingester
"""
from __future__ import annotations

import sqlite3
from typing import Iterable

import pandas as pd

from pitgenius.config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS races (
    year        INTEGER NOT NULL,
    round       INTEGER NOT NULL,
    event_name  TEXT NOT NULL,
    country     TEXT,
    location    TEXT,
    circuit     TEXT,
    date        TEXT,
    PRIMARY KEY (year, round)
);

CREATE TABLE IF NOT EXISTS laps (
    year            INTEGER NOT NULL,
    round           INTEGER NOT NULL,
    driver          TEXT NOT NULL,
    driver_id       TEXT,
    team            TEXT,
    lap_number      INTEGER NOT NULL,
    lap_time_s      REAL,
    sector1_s       REAL,
    sector2_s       REAL,
    sector3_s       REAL,
    tire_compound   TEXT,
    tire_life       INTEGER,   -- laps on current set (stint age)
    stint_number    INTEGER,
    is_pit_out_lap  INTEGER,   -- 0/1
    track_status    TEXT,      -- FastF1 TrackStatus string e.g. '1','2','45'
    position        INTEGER,
    PRIMARY KEY (year, round, driver, lap_number)
);

CREATE TABLE IF NOT EXISTS pit_stops (
    year        INTEGER NOT NULL,
    round       INTEGER NOT NULL,
    driver      TEXT NOT NULL,
    stop_number INTEGER NOT NULL,
    lap         INTEGER,
    duration_s  REAL,          -- total pit dwell time per F1 timing
    pit_time_s  REAL,          -- stationary time if available
    PRIMARY KEY (year, round, driver, stop_number)
);

CREATE TABLE IF NOT EXISTS results (
    year            INTEGER NOT NULL,
    round           INTEGER NOT NULL,
    driver          TEXT NOT NULL,
    team            TEXT,
    position        INTEGER,   -- NULL = DNF/DNS/DSQ
    status          TEXT,      -- 'Finished', '+1 Lap', engine failure, ...
    points          REAL,
    grid_position   INTEGER,
    total_laps      INTEGER,
    PRIMARY KEY (year, round, driver)
);

CREATE TABLE IF NOT EXISTS ingestion_manifest (
    year         INTEGER NOT NULL,
    round        INTEGER NOT NULL,
    session_code TEXT NOT NULL,
    loaded_at    TEXT NOT NULL,
    n_rows_laps  INTEGER,
    status       TEXT NOT NULL DEFAULT 'ok',
    error        TEXT,
    PRIMARY KEY (year, round, session_code)
);
"""


def connect(db_path=DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    return conn


def write_dataframe(df: pd.DataFrame, table: str, conn: sqlite3.Connection,
                    if_exists: str = "append") -> int:
    """Write a DataFrame to SQLite, replacing existing rows keyed by its PKs.

    For simplicity and idempotency we delete rows matching this df's
    (year, round) scope before inserting.
    """
    if df.empty:
        return 0
    cols = set(pd.read_sql_query(f"SELECT * FROM {table} LIMIT 0", conn).columns)
    df = df[[c for c in df.columns if c in cols]]
    if "year" in df.columns and "round" in df.columns:
        years = df["year"].unique().tolist()
        rounds = df["round"].unique().tolist()
        q = ",".join("?" * len(years))
        r = ",".join("?" * len(rounds))
        conn.execute(
            f"DELETE FROM {table} WHERE year IN ({q}) AND round IN ({r})",
            years + rounds,
        )
    df.to_sql(table, conn, if_exists=if_exists, index=False)
    return len(df)


def load_laps(year: int | None = None, round_: int | None = None,
              conn: sqlite3.Connection | None = None) -> pd.DataFrame:
    return _load("laps", year, round_, conn)


def load_races(conn: sqlite3.Connection | None = None) -> pd.DataFrame:
    return _load("races", None, None, conn)


def load_results(year: int | None = None, round_: int | None = None,
                 conn: sqlite3.Connection | None = None) -> pd.DataFrame:
    return _load("results", year, round_, conn)


def load_pit_stops(year: int | None = None, round_: int | None = None,
                   conn: sqlite3.Connection | None = None) -> pd.DataFrame:
    return _load("pit_stops", year, round_, conn)


def load_manifest(conn: sqlite3.Connection | None = None) -> pd.DataFrame:
    return _load("ingestion_manifest", None, None, conn)


def _load(table: str, year, round_, conn) -> pd.DataFrame:
    own = conn is None
    c = conn or connect()
    try:
        q = f"SELECT * FROM {table}"
        conds, params = [], []
        if year is not None:
            conds.append("year = ?"); params.append(year)
        if round_ is not None:
            conds.append("round = ?"); params.append(round_)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        return pd.read_sql_query(q, c, params=params)
    finally:
        if own:
            c.close()


def already_ingested(year: int, round_: int, session_code: str,
                     conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT status FROM ingestion_manifest WHERE year=? AND round=? AND session_code=?",
        (year, round_, session_code),
    ).fetchone()
    return bool(row and row[0] == "ok")


def record_manifest(year: int, round_: int, session_code: str,
                    n_rows_laps: int, status: str = "ok", error: str | None = None,
                    conn: sqlite3.Connection | None = None):
    own = conn is None
    c = conn or connect()
    try:
        c.execute(
            "INSERT OR REPLACE INTO ingestion_manifest "
            "(year, round, session_code, loaded_at, n_rows_laps, status, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (year, round_, session_code,
             pd.Timestamp.utcnow().isoformat(), n_rows_laps, status, error),
        )
        c.commit()
    finally:
        if own:
            c.close()