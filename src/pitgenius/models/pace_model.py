"""Pre-race car/driver pace offsets (DECISIONS.md D23).

HONEST DEFINITION - not a black-box "team strength" score:

    pace_offset(race, driver) = field_median(fuel-adjusted clean lap)
                                - driver_median(fuel-adjusted clean lap)

Positive = driver ran FASTER than the field median that race. Lap times are
fuel-adjusted with the project-wide D14 proxy (FUEL_S_PER_LAP), reusing the
existing convention rather than re-deriving anything.

LEAKAGE RULE (the whole point of this module): only ROLLING, PRE-RACE
estimates may be fed into predictors. RollingPaceModel.offsets_before()
returns, per driver, the mean of that driver's pace offsets from the last
n_recent races STRICTLY BEFORE the target race. Same-race outcome data is
never used to estimate same-race pace - the same temporal discipline as
D22's calibration layer.

Chronology is (year, round) ordering; rounds are calendar-ordered within a
season, so this is a faithful race sequence. Drivers without enough prior
history have no entry - callers must handle absence explicitly (fill value
is the caller's policy, never silently zero here).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pitgenius.analysis.degradation import green_flag_laps
from pitgenius.models.degradation_model import FUEL_S_PER_LAP

MIN_CLEAN_LAPS = 5      # below this a race's offset is not estimated
DEFAULT_N_RECENT = 5


def race_pace_offsets(laps: pd.DataFrame) -> pd.DataFrame:
    """Per-(year, round, driver) pace offset in s/lap (positive = faster).

    Fuel-adjusted lap time = lap_time_s - FUEL_S_PER_LAP * remaining_laps,
    i.e. normalized to zero fuel load (D14 proxy).
    """
    gf = green_flag_laps(laps)
    if gf.empty:
        return pd.DataFrame(
            columns=["year", "round", "driver", "team", "pace_offset_s"])
    total = gf.groupby(["year", "round"])["lap_number"].transform("max")
    gf = gf.assign(
        _base=gf["lap_time_s"] - FUEL_S_PER_LAP * (total - gf["lap_number"]))
    grp = ["year", "round", "driver"]
    n_laps = (gf.groupby(grp)["_base"].size().rename("n_clean_laps")
              .reset_index())
    med = (gf.groupby(grp)["_base"].median().rename("_base_med")
           .reset_index())
    field = (gf.groupby(["year", "round"])["_base"].median()
             .rename("_field").reset_index())
    team = (gf.sort_values("lap_number").groupby(grp)["team"].last()
            .rename("team").reset_index())

    out = med.merge(field, on=["year", "round"], how="left")
    out = out.merge(n_laps, on=grp, how="left")
    out = out.merge(team, on=grp, how="left")
    out["pace_offset_s"] = out["_field"] - out["_base_med"]
    out = out[out["n_clean_laps"] >= MIN_CLEAN_LAPS]
    return out[["year", "round", "driver", "team", "n_clean_laps",
                "pace_offset_s"]].reset_index(drop=True)


class RollingPaceModel:
    """Pre-race pace estimates from each driver's recent prior races."""

    def __init__(self, laps: pd.DataFrame, n_recent: int = DEFAULT_N_RECENT):
        self.n_recent = n_recent
        offs = race_pace_offsets(laps)
        races = (offs[["year", "round"]].drop_duplicates()
                 .sort_values(["year", "round"]).reset_index(drop=True))
        self._pos = {}
        for i, r in races.iterrows():
            self._pos[(int(r["year"]), int(r["round"]))] = i
        self._hist: dict[str, list[tuple[int, float]]] = {}
        for _, row in offs.iterrows():
            key = (int(row["year"]), int(row["round"]))
            if key not in self._pos:
                continue  # race had < MIN_CLEAN_LAPS rows somewhere upstream
            self._hist.setdefault(row["driver"], []).append(
                (self._pos[key], float(row["pace_offset_s"])))
        for d in self._hist:
            self._hist[d].sort()

    def offsets_before(self, year: int, round_: int) -> dict[str, float]:
        """{driver: mean offset over last n_recent races before this one}."""
        pos = self._pos.get((int(year), int(round_)))
        if pos is None:
            # Race itself unranked (e.g. too few clean laps): treat every
            # known race as 'before' rather than returning nothing.
            pos = 10 ** 9
        out = {}
        for driver, hist in self._hist.items():
            prior = [o for p, o in hist if p < pos]
            if prior:
                out[driver] = float(np.mean(prior[-self.n_recent:]))
        return out

    def driver_offset_before(self, year: int, round_: int,
                             driver: str) -> float | None:
        return self.offsets_before(year, round_).get(driver)
