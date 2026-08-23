"""FastF1 ingestion pipeline: pull race sessions, clean, store.

Resumable and idempotent (DECISIONS.md D5): every (year, round, session)
already recorded 'ok' in the manifest is skipped. Failures are logged to the
manifest and skipped so a bad session never kills a long backfill.

Usage:
    python -m src.pitgenius.data.ingest --seasons 2024 2025
    python -m src.pitgenius.data.ingest --seasons 2021 2022 2023 2024 2025
    python -m src.pitgenius.data.ingest --seasons 2025 --rounds 10
"""
from __future__ import annotations

import argparse
import logging
import time

import fastf1
import pandas as pd

from pitgenius.config import CACHE_DIR, RAW_DIR, RACE_SESSION_CODE
from pitgenius.data import clean, store

log = logging.getLogger("pitgenius.ingest")

# F1 livetiming API rate limits; back off and retry rather than fail.
RATE_LIMIT_MARKERS = ("500 calls/h", "429", "Too Many Requests")
RATE_LIMIT_SLEEP_S = 120
MAX_RETRIES = 5


def configure_cache(cache_dir=CACHE_DIR):
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def ingest_race(year: int, round_: int, session_code: str = RACE_SESSION_CODE,
                conn=None) -> dict:
    """Ingest one (year, round, session). Returns a summary dict."""
    summary = {"year": year, "round": round_, "session": session_code,
               "status": "ok", "laps": 0}
    t0 = time.time()
    session = fastf1.get_session(year, round_, session_code)
    session.load(laps=True, telemetry=False, weather=False, messages=True)

    meta = clean.clean_race_meta(session)
    laps_df = clean.clean_laps(session)
    pits_df = clean.clean_pit_stops(session)
    results_df = clean.clean_results(session)

    if laps_df.empty or results_df.empty:
        raise ValueError(f"empty laps/results for {year} R{round_}")

    store.write_dataframe(pd.DataFrame([meta]), "races", conn)
    n_laps = store.write_dataframe(laps_df, "laps", conn)
    store.write_dataframe(pits_df, "pit_stops", conn)
    store.write_dataframe(results_df, "results", conn)
    store.record_manifest(year, round_, session_code, n_laps,
                          status="ok", conn=conn)
    conn.commit()

    # Parquet dump for analysis convenience.
    out = RAW_DIR / f"{year}_r{round_:02d}_{session_code.lower()}.parquet"
    laps_df.to_parquet(out, index=False)

    summary["laps"] = n_laps
    summary["seconds"] = round(time.time() - t0, 1)
    return summary


def backfill(seasons, rounds=None, session_code=RACE_SESSION_CODE,
             db_path=None) -> pd.DataFrame:
    """Ingest all requested seasons; skip manifest hits; log failures."""
    conn = store.connect(db_path) if db_path else store.connect()
    results = []
    try:
        for year in seasons:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            today = pd.Timestamp.utcnow().tz_localize(None)
            for _, ev in schedule.iterrows():
                if ev["EventDate"] > today:
                    continue  # not yet run
                round_ = int(ev["RoundNumber"])
                if rounds and round_ not in rounds:
                    continue
                if store.already_ingested(year, round_, session_code, conn):
                    log.info("skip %d R%d (already ingested)", year, round_)
                    continue
                attempt = 0
                while True:
                    try:
                        s = ingest_race(year, round_, session_code, conn)
                        results.append(s)
                        log.info("ok   %d R%-2d %-25s laps=%s in %.0fs",
                                 year, round_, ev["EventName"], s["laps"],
                                 s.get("seconds", 0))
                        break
                    except Exception as exc:  # noqa: BLE001 — must survive
                        msg = str(exc)
                        if any(mk in msg for mk in RATE_LIMIT_MARKERS) \
                                and attempt < MAX_RETRIES:
                            attempt += 1
                            log.warning("rate-limited at %d R%d; sleeping "
                                        "%ds (attempt %d/%d)", year, round_,
                                        RATE_LIMIT_SLEEP_S, attempt,
                                        MAX_RETRIES)
                            time.sleep(RATE_LIMIT_SLEEP_S)
                            continue
                        log.warning("FAIL %d R%d: %s", year, round_, exc)
                        store.record_manifest(year, round_, session_code, 0,
                                              status="error",
                                              error=msg[:500], conn=conn)
                        conn.commit()
                        results.append({"year": year, "round": round_,
                                        "status": "error", "error": msg})
                        break
    finally:
        conn.close()
    return pd.DataFrame(results)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="PitGenius FastF1 ingestion")
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    ap.add_argument("--rounds", type=int, nargs="*", default=None,
                    help="optional specific rounds")
    args = ap.parse_args()

    configure_cache()
    df = backfill(args.seasons, rounds=args.rounds)
    ok = (df["status"] == "ok").sum() if len(df) else 0
    err = (df["status"] == "error").sum() if len(df) else 0
    print(f"Done. ingested={ok} errors={err}")
    if len(df):
        print(df.to_string())


if __name__ == "__main__":
    main()