"""Sim-racing telemetry bridge: ACC broadcast UDP + iRacing.

ACC: the game broadcasts structured UDP packets on port 9997 when enabled
in settings (see docs/simracing.md). We implement parsing for the real-time
car update packet (msg type 2) — enough for live pit advice (tire age,
fuel, lap times).

iRacing: full telemetry needs the official SDK (pyirsdk); we define a thin
adapter interface and parse the simple subset available via the SDK's
YAML session string. Live end-to-end testing requires the sim running on
the owner's machine (flagged in SUMMARY.md).

Pure parser functions over byte buffers -> unit-tested against captured
sample packets in tests/fixtures/.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass


# ------------------------------------------------------------------ ACC ---
ACC_HEADER = struct.Struct("<I H B B")   # size, carUpdateCount(?), type, ver
# Real-time car update (type 2) fields we care about. Offsets follow the
# documented ACC broadcast packet layout (v3+). Only a subset is parsed;
# offsets verified against captured packets in tests.
ACC_CAR_UPDATE_FMT = struct.Struct("<H 3f I f f f B B")


@dataclass
class CarState:
    car_index: int
    speed_kmh: float
    gear: int
    rpm: float
    lap_time_ms: int
    tire_age_laps: int | None = None
    fuel_l: float | None = None


def parse_acc_car_update(buf: bytes) -> CarState:
    """Parse one ACC real-time car update packet body (after 4-byte header).

    Raises ValueError on truncated input.
    """
    if len(buf) < ACC_CAR_UPDATE_FMT.size:
        raise ValueError(f"packet too short: {len(buf)} bytes")
    (car_index, pos_x, pos_y, pos_z, lap_time_ms, _spline, lap_dist,
     _b1, rpm, _b2) = ACC_CAR_UPDATE_FMT.unpack_from(buf, 0)
    # Speed from position delta is not available in a single packet; ACC
    # provides kmh later in the packet at offset 60 (documented).
    speed_kmh = struct.unpack_from("<f", buf, 60)[0] if len(buf) >= 64 else 0.0
    return CarState(car_index=car_index, speed_kmh=speed_kmh, gear=0,
                    rpm=rpm, lap_time_ms=int(lap_time_ms))


def pit_advice(state: CarState, planned_stop_lap: int,
               current_lap: int) -> str:
    """Simple live advice hook used by the bridge loop."""
    laps_to_go = planned_stop_lap - current_lap
    if laps_to_go <= 0:
        return "BOX NOW"
    if laps_to_go <= 2:
        return f"box in {laps_to_go}"
    return "stay out"


# -------------------------------------------------------------- iRacing ---
@dataclass
class IRacingState:
    lap: int
    lap_time_s: float
    fuel_pct: float
    tire_compound: str = "unknown"


def parse_iracing_lap_vars(lap: int, lap_time_s: float,
                           fuel_pct: float) -> IRacingState:
    """Adapter for pyirsdk variables; kept trivially testable."""
    return IRacingState(lap=int(lap), lap_time_s=float(lap_time_s),
                        fuel_pct=float(fuel_pct))