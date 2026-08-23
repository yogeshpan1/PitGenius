# Sim-Racing Bridge — how to test live on your machine

The parsers are unit-tested against synthetic packets (`tests/test_core.py`),
but **live end-to-end testing requires the sim actually running** on your
machine — that part cannot be done from this build session.

## ACC (Assetto Corsa Competizione)

1. Enable the broadcast server in ACC:
   `Documents\Assetto Corsa Competizione\Config\broadcasting.json`
   - set `updListenerPort` (default 9997) and a `connectionPassword`.
2. Run the bridge loop (see `src/pitgenius/simracing/bridge.py`) pointed at
   `127.0.0.1:9997`.
3. Verify parsed CarState values against the in-game HUD for one stint;
   adjust offsets if your game version's packet layout differs (the struct
   layout is documented at the top of bridge.py and was validated against
   the v3+ broadcast spec).

## iRacing

Full telemetry needs `pyirsdk` (`pip install pyirsdk`). The adapter in
`bridge.py` takes lap/fuel variables directly; connect via
`irsdk.IRSDK()` → `startup()` → read `Lap`, `LapLastLapTime`, `FuelLevel`.

## Feeding pit advice

`pit_advice(state, planned_stop_lap, current_lap)` returns BOX NOW /
box-in-N / stay out. Plug planned_stop_lap from the Monte Carlo simulator's
recommended strategy for the current track state.