# PitGenius — Decision Log

This file records every genuinely ambiguous implementation decision made during
the build, plus the reasoning. It is appended to as the project grows.

---

## D1 — Data storage: SQLite + Parquet, not a full DBMS
**Decision:** Cleaned race data is stored in a single SQLite database
(`data/pitgenius.db`) for queryable access, with per-season Parquet dumps in
`data/raw/` as the analysis-friendly interchange format.
**Reasoning:** Single-user analytics workload, local file, zero ops overhead.
SQLite ships with Python; Parquet preserves dtypes (critical for timedelta
columns like lap times) that SQLite cannot represent natively. If this ever
needs multi-user/concurrent writes (live weekend ingestion + dashboard reads),
the migration path is Postgres — documented, not built.

## D2 — Lap times stored as numeric seconds, not timedeltas
**Decision:** All durations (lap time, sector times, pit lane time) are stored
as float seconds in SQLite/Parquet. Nulls preserved as NaN.
**Reasoning:** SQLite has no duration type; scikit-learn needs floats anyway;
conversion to `m:ss.mmm` happens only at presentation time.

## D3 — "Last 5 complete seasons" = 2021–2025
**Decision:** Ingestion targets seasons 2021 through 2025 inclusive (2026 is
in progress at build time, so it is *not* a complete season).
**Reasoning:** Literal reading of the brief. The 2026 season, once complete,
can be added by re-running the same ingestion command.

## D4 — Ingestion scope per session: Race sessions only for the full grid
**Decision:** For each historical event we ingest the **Race** session (`R`)
only: laps, tire compounds/stints, pit stops, and final results. Qualifying
and practice sessions are out of scope for v1.
**Reasoning:** Every model in the plan (degradation, undercut, Monte Carlo,
backtests) consumes race data. Pulling quali/practice for ~110 events would
roughly triple network I/O time for no current modeling benefit. The
ingestion module takes a session-code parameter so extending later is a
one-flag change.

## D5 — Resumable, idempotent ingestion with a manifest table
**Decision:** The ingester keeps an `ingestion_manifest` table recording which
(year, round, session) triples are already loaded, and skips them on re-run.
Failures are logged and skipped, not fatal.
**Reasoning:** FastF1 pulls are slow and network-bound; a crash mid-backfill
must not restart from zero. Idempotency makes "top up with new races" a
single command.

## D6 — Quantile regression via GradientBoosting + LightGBM quantile objective
**Decision:** P10/P50/P90 lap-time prediction uses LightGBM's native quantile
objective (three models, one per quantile), with a scikit-learn
`HistGradientBoostingRegressor(loss="quantile")` fallback baseline.
**Reasoning:** Native, fast, handles non-linear tire cliff behavior, and gives
well-calibrated intervals when validated with pinball loss — which we do.

## D7 — Honesty standard for accuracy claims
**Decision:** Every reported metric states (a) the exact race set it was
computed on, (b) whether circuits/races were held out from training, and (c)
failure counts alongside success counts. No aggregate-only reporting.
**Reasoning:** Direct application of the owner's prior experience: 94%
cross-validated vs 62.5% on independently written real-world phrases. CV
numbers alone systematically overstate real-world performance; temporal and
per-circuit splits are mandatory here because F1 data is strongly
autocorrelated within a season.

## D8 — LLM explainability behind an interface with a deterministic stub
**Decision:** The explanation layer calls an OpenAI-compatible chat API
(Groq-compatible) using `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` env vars.
If no key is present, a clearly-labelled deterministic template stub returns
the explanation instead, marked `STUB` in its output.
**Reasoning:** No credentials available during the build. The interface means
supplying a key later requires zero code changes. Confidence gating lives in
code that runs regardless of backend, so gating logic is testable today.

## D9 — Live predictions: append-only JSON files as the immutable store
**Decision:** Published predictions are written once to
`predictions/<season>/<round>/pre_race.json` with a UTC timestamp and SHA-256
content hash; the scoring step never modifies them, it only writes a sibling
`post_race_score.json`. A hash mismatch on re-score raises an error.
**Reasoning:** Git itself provides auditability, but the hash makes tampering
detectable even outside git. A database adds nothing for a file-per-race
artifact that must be publicly reviewable.

## D10 — Sim-racing bridge: pure parser functions + recorded sample packets
**Decision:** ACC (UDP broadcast) and iRacing (SDK-style) telemetry parsing is
implemented as pure functions over byte buffers, unit-tested against captured
sample packets checked into `tests/fixtures/`. Live socket loops exist but
cannot be end-to-end tested without the sim running.
**Reasoning:** Per the brief: live sim telemetry only exists on the owner's
machine with the game running. Pure parsers give us everything testable now;
the socket layer is thin and flagged for live testing.

## D11 — Frontend sequencing: Streamlit first, React second
**Decision:** Streamlit dashboard ships with Stage 2 for immediate analytical
value. The production React frontend (Stage 8) talks to the FastAPI backend.
**Reasoning:** Matches the brief's stack plan; Streamlit unblocks validation
work immediately while React/API work proceeds in parallel stages.

## D12 — Safety car modeling: empirical per-circuit rate + Poisson-ish sampling
**Decision:** SC probability per circuit is estimated from historical
frequency of safety-car / virtual-safety-car laps per race at that circuit,
with shrinkage toward the global mean for circuits with few samples. The
simulator samples incident counts rather than binary SC/no-SC.
**Reasoning:** Small-sample frequencies are noisy; shrinkage is the honest
simple fix. Count-based sampling captures multiple-SC races which binary
sampling misses.

## D14 — Fuel-load handling: linear proxy, not pseudo-precision
**Decision:** Lap-time models include a linear fuel proxy of 0.033 s/lap
decreasing from race start, applied identically across all models. No claim
is made that this is a true fuel-mass correction.
**Reasoning:** Fuel mass per lap is not public; the ~0.033 s/lap figure is the
commonly cited order of magnitude. What matters for strategy comparisons is
that the same correction is applied to every stint, so *relative* degradation
between compounds/strategies is consistent. Claiming a fancier correction
would be fake precision.

## D13 — Backtest protocol: leave-one-race-out within season + full-season holdout
**Decision:** Degradation-model backtests report both (a) leave-one-race-out
across all ingested races and (b) train-on-seasons-A/B → test-on-season-C.
The harder (b) number is treated as the headline honest estimate.
**Reasoning:** (a) measures interpolation skill; (b) measures the actual
deployment scenario (predicting future seasons). Reporting only (a) would be
exactly the 94%-vs-62.5% trap.