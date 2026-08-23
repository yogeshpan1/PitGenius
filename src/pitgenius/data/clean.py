"""Transform raw FastF1 session objects into clean, model-ready DataFrames.

All durations are float seconds (DECISIONS.md D2). Cleaning rules:
- Lap times: drop nulls, drop absurd outliers (< 60s or > 150s — these are
  pit in/out laps, formation anomalies, or timing glitches).
- Tire compound: normalize FastF1 compound names (SOFT/MEDIUM/HARD/...);
  UNKNOWN / NaN kept as explicit 'UNKNOWN'.
- Stint/tire-life reconstruction: FastF1 provides TyreLife; where missing we
  reconstruct from Stint number boundaries.
- Track status preserved so analysis can filter green-flag vs SC laps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VALID_COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
# Plausible race-lap-time window in seconds (shortest F1 lap ~ 64s at
# A1-Ring-style laps; slowest legit wet laps well under 150s).
MIN_PLAUSIBLE_LAP_S = 55.0
MAX_PLAUSIBLE_LAP_S = 150.0


def _td_to_seconds(series: pd.Series) -> pd.Series:
    """FastF1 returns timedeltas (NaT-aware); convert to float seconds."""
    return pd.to_numeric(series.dt.total_seconds(), errors="coerce")


def clean_laps(session) -> pd.DataFrame:
    """Build the clean laps DataFrame from a loaded FastF1 Session."""
    laps = session.laps.pick_accurate().copy()
    if laps.empty:
        return pd.DataFrame()

    df = pd.DataFrame({
        "year": int(session.event.year),
        "round": int(session.event.RoundNumber),
        "driver": laps["Driver"],
        "team": laps.get("Team"),
        "lap_number": laps["LapNumber"].astype(int),
        "lap_time_s": _td_to_seconds(laps["LapTime"]),
        "sector1_s": _td_to_seconds(laps["Sector1Time"]),
        "sector2_s": _td_to_seconds(laps["Sector2Time"]),
        "sector3_s": _td_to_seconds(laps["Sector3Time"]),
        "tire_compound": laps.get("Compound"),
        "tire_life": pd.to_numeric(laps.get("TyreLife"), errors="coerce"),
        "stint_number": pd.to_numeric(laps.get("Stint"), errors="coerce"),
        "is_pit_out_lap": laps.get("PitOutTime").notna().astype(int),
        "track_status": laps.get("TrackStatus"),
        "position": pd.to_numeric(laps.get("Position"), errors="coerce"),
    })

    df["tire_compound"] = (
        df["tire_compound"].fillna("UNKNOWN").str.upper().where(
            df["tire_compound"].fillna("UNKNOWN").str.upper().isin(VALID_COMPOUNDS),
            "UNKNOWN",
        )
    )

    # Reconstruct tire_life where missing: lap - first lap of stint + 1.
    need = df["tire_life"].isna()
    if need.any():
        first_lap_of_stint = df.groupby(["driver", "stint_number"])[
            "lap_number"].transform("min")
        reconstructed = df["lap_number"] - first_lap_of_stint + 1
        df.loc[need, "tire_life"] = reconstructed[need]

    # Drop implausible lap times but keep the row (sector data may still be
    # useful); models filter on lap_time_s.notna().
    bad = (df["lap_time_s"] < MIN_PLAUSIBLE_LAP_S) | \
          (df["lap_time_s"] > MAX_PLAUSIBLE_LAP_S)
    df.loc[bad, "lap_time_s"] = np.nan

    return df


def clean_pit_stops(session) -> pd.DataFrame:
    """Build clean pit-stop rows from a loaded FastF1 Session."""
    laps = session.laps.copy()
    if laps.empty:
        return pd.DataFrame()

    pits = laps[laps["PitInTime"].notna()][["Driver", "LapNumber", "PitInTime"]]
    rows = []
    for driver, grp in pits.groupby("Driver"):
        grp = grp.sort_values("LapNumber")
        for i, (_, r) in enumerate(grp.iterrows(), start=1):
            rows.append({
                "year": int(session.event.year),
                "round": int(session.event.RoundNumber),
                "driver": driver,
                "stop_number": i,
                "lap": int(r["LapNumber"]),
                # FastF1 laps table has no stationary time; duration comes
                # from the pit_stop timing API below when available.
                "duration_s": np.nan,
                "pit_time_s": np.nan,
            })
    df = pd.DataFrame(rows)

    # Enrich with official pit-stop durations if present on the session.
    try:
        off = session.pit_stops  # requests-based timing data
        if off is not None and len(off):
            dur = {}
            stat = {}
            for _, r in off.iterrows():
                key = (r.get("Driver", r.get("driver")), int(r["Lap"]))
                d = r.get("Duration")
                dur[key] = pd.to_timedelta(d).total_seconds() if d is not None else np.nan
            if len(df):
                keys = list(zip(df["driver"], df["lap"]))
                df["duration_s"] = [dur.get(k, np.nan) for k in keys]
    except Exception:
        pass  # pit stop enrichment is best-effort

    return df


def clean_results(session) -> pd.DataFrame:
    """Build clean final-classification rows from a loaded FastF1 Session."""
    res = session.results.copy()
    if res.empty:
        return pd.DataFrame()

    def _status(r):
        for col in ("Status", "status"):
            if col in res.columns and pd.notna(r.get(col)):
                return str(r[col])
        return "Unknown"

    df = pd.DataFrame({
        "year": int(session.event.year),
        "round": int(session.event.RoundNumber),
        "driver": res["Abbreviation"],
        "team": res.get("TeamName", res.get("Team")),
        "position": pd.to_numeric(res["Position"], errors="coerce"),
        "status": res.apply(_status, axis=1),
        "points": pd.to_numeric(res.get("Points"), errors="coerce"),
        "grid_position": pd.to_numeric(res.get("GridPosition"), errors="coerce"),
        "total_laps": pd.to_numeric(res.get("Laps"), errors="coerce"),
    })
    # Position 0 or NaN means not classified (DNF/DNS/DSQ) -> keep NULL.
    df.loc[df["position"] <= 0, "position"] = np.nan
    return df


def clean_race_meta(session) -> dict:
    ev = session.event
    return {
        "year": int(ev.year),
        "round": int(ev.RoundNumber),
        "event_name": str(ev.EventName),
        "country": str(ev.Country),
        "location": str(ev.Location),
        "circuit": str(getattr(ev, "Location", "")),
        "date": str(ev.EventDate.date()),
    }