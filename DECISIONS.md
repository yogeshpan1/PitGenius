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
## D20 - Quantile coverage below target is reported as-is  
First full validation run: P10-P90 coverage 61.7%% (LORO) / 68.4%% (2025 holdout) vs 80%% target. LightGBM quantile models under-cover on noisy lap-time deltas. NOT widened post-hoc to flatter the numbers; documented in reports/degradation_validation.json and methodology section 7. Future fix candidates: conformal calibration layer, wider feature set (weather), per-circuit interval scaling.  
  
## D21 - Circuit identity keyed by (round, year)  
Each race is its own categorical so the model learns per-race baselines from other drivers in the same race (the deployment scenario). Downside: no cross-year circuit transfer; accepted because calendar drift makes round-number identity unreliable anyway.  
  
## D22 - Undercut calculator FAILED its backtest; reported honestly  

## D22 - Undercut calculator FAILED its backtest; reported honestly

First full backtest: Brier 0.714 vs constant-baseline 0.248 and kind-baseline ~0.19. The calculator is ANTI-correlated with outcomes on detected attempts. Causes: (a) degradation model gives implausible fresh-tire deltas (-3s vs driver median at age 2) because the delta-vs-own-median target conflates tire state with lap context at stint starts; (b) attempt detection is endogenous - teams attempt undercuts precisely when they work, so within the detected sample the gap carries little signal (undercut base rate 96 pct, overcut 2 pct). Next steps: fix the degradation target (per-stint reference lap instead of race median), model P(attempt) separately from P(success|attempt), evaluate on ALL adjacent-stop pairs not just detected attempts.

### D22 UPDATE - all three fixes implemented and re-evaluated; still negative, reported in full

Fix 1 (degradation target): build_training_frame now uses a per-stint, fuel-adjusted reference (median clean lap over the first 3 tire-life laps of the SAME stint; fuel-adjusted via the D14 0.033 s/lap proxy because verification exposed the raw stint reference as a FALLING age curve from within-stint fuel burn-off). Probe (scripts/verify_d22_target.py, trained <2022): fresh-tire P50 delta at age 2 went -1.759s -> +0.751s; curve now rises to a ~+1.4s plateau. 21,366/93,610 clean laps have no stint reference and are dropped (counted, documented). On the OLD detected-attempts protocol this alone moved Brier 0.7141 -> 0.7590: no improvement, as expected while the sample stays selected.

Fix 2 (P(attempt) separate): undercut_history.find_adjacent_pairs builds ALL adjacent-stop pairs canonically (attacker = first pitter, one record per event; is_attempt = legacy <=2-lap response). strategy/attempt_model.py is a logistic P(respond | gap, ages, compound diff), trained temporally. Holdout result: Brier 0.2378 vs base-rate 0.2361, ROC-AUC 0.532 - the features carry almost no attempt signal. Reported as-is.

Fix 3 (all-pairs evaluation): scripts/backtest_undercut.py scores every pair. Selection bias removed from EVALUATION as designed: first-pitter retention base rate drops from ~96 pct (selected attempts) to ~15 pct (all pairs). Result: PRIMARY all pairs n=1852: Brier(model)=0.5005 vs constant-baseline 0.1383 and kind-baseline 0.1256; LogLoss(model)=4.3626. Subset is_attempt n=1144: 0.5076 vs 0.1482.

VERDICT (superseded by D22 fixes 4+5 below): the D22 mechanics are implemented and measured correctly, but the calculator STILL does not beat naive baselines - its p_flip clusters at extremes while outcomes sit near 15 pct. Remaining diagnosed defect: the physics p_flip's calibration/orientation itself (evaluate_move's kind/gap sign conventions are suspect and unvalidated against outcomes). Do not trust undercut probabilities.

### D22 fixes 4+5 - orientation bug found and fixed; calibration layer added

Diagnosis (scripts/audit_pflip_orientation.py): v1-v3 returned P(gap_after<0), i.e. the probability the FIRST PITTER LOSES, but scored it against labels where success means the first pitter RETAINS. Real-data audit: corr(p_predicted, success) = -0.328 - predictions were literally inverted since v1; this, not selection bias alone, is why every earlier protocol looked anti-correlated.

Fix 4: evaluate_move rewritten - returns p_success = P(attacker ahead after both stops) matching the label, and models multi-lap physics (response_laps: k laps of responder old-tire degradation vs attacker fresh-tire laps; v1-v3 modeled one lap and ignored k). recommend() reframed for the trailing car. Tests updated to retention semantics + multi-lap monotonicity test (16 passed).

Fix 5: temporal calibration layer in scripts/backtest_undercut.py - logistic fitted only on prior-season pairs (raw physics p, log-gap capped 60s, kind, response_laps), applied to each test season. Raw-physics scores reported alongside.

Verification chain (all real runs):
- Orientation+multi-lap only (uncalibrated): Brier 0.5005 -> 0.3588, LogLoss 4.3626 -> 2.4807.
- With calibration: Brier(calibrated)=0.1295 (first feature set); adding capped gap + response_laps gave 0.1298 - second iteration changed nothing material, stopped tuning.
- Fair baselines (scripts/audit_baselines.py, n-weighted pooled): model 0.1295 vs constant base-rate 0.1373 (BEATEN, both Brier and log loss 0.4115 vs 0.4491) vs non-leaky prior-kind-rates baseline 0.1270 (NOT beaten, gap ~0.003) vs oracle kind-rates fitted on test season 0.1235.
- Per season: model beats constant baseline in 2024 (0.1258 vs 0.1422) and 2025 (0.1311 vs 0.1643); loses 2022/2023 where the calibrator has least history.

STATUS: first time this component beats naive constant-rate guessing out of sample. D22 remains OPEN by the project bar: it must also beat trivial prior-information baselines (per-kind prior-season rates); remaining gap ~0.003 Brier most plausibly needs a car-pace/dominance feature (same root cause as the simulator's D20 problem). Do not treat probabilities as production-grade until that lands.

BoxBox cross-check (scripts/crosscheck_boxbox.py, real output in reports/_boxbox_crosscheck.log): pit_strategy.csv labels usable on only 6/1388 rows (0.4 pct; strategy_outcome is "Unknown" elsewhere and derives from FastF1's noisy Position lag/lead, with no adjacency or rival comparison) - EXCLUDED from D22 entirely. team_strategy.csv is a dated 2021-22 proxy whose success counts inherit those broken labels (mean "undercut success" 1.8 pct vs our same-era detected rates: 92.5 pct undercut / 4.6 pct overcut) - partial, dated signal only, relevant at most to the simulator dominance term later. Per-lap CSVs: all_race_laps_Final.csv (45,115 laps, 42 races) already contains BOTH 2021 AND 2022 - merging it with 2021_remaining.csv and all_race_laps_2022_complete.csv would double-count; era overlaps the ingested DB (108 races, 2021-2025), so boxbox laps are redundant for D22 and add only sector/speed-trap columns.

## D23 - Car/team pace model: built, validated, wired into both downstream models

MOTIVATION: D22's residual gap vs prior-kind baselines and the simulator's 0/62 band failure shared one root cause - nothing conditioned on which car is simply faster.

PART 1 - pace model (src/pitgenius/models/pace_model.py): honest definition, per-driver offset = field median minus driver median of FUEL-ADJUSTED clean laps (D14 proxy reused). Pre-race estimate = mean of last 5 prior races' offsets (offsets_before); same-race data never used for same-race pace. Validation (scripts/validate_pace_model.py): coverage 98 pct of driver-races; Spearman(pre-race pace, finish) = +0.594 (0.53-0.67 every season); reproduces known eras from pre-race data only - Red Bull rank 1/10 teams in dominant 2023 (+0.814 s/lap), McLaren top 2024-25, Haas -1.47 in 2021.

PART 2 - undercut calibrator + pace features: NO improvement. Brier 0.1317 with pace vs 0.1295 without; non-leaky kind-prior baseline 0.1270 still not beaten. Interpretation: adjacent-stop outcomes are decided in-race; the cumulative-gap feature already encodes relative pace, and season-level rolling pace is too coarse here. D22's kind-prior gap remains OPEN.

PART 3 - simulator dominance term: MonteCarloSimulator accepts per-car pace offsets (added to lap time); backtest_race feeds real offsets (winner focal gets winner's pace; candidate focals field-median; field cars their own). Two wiring bugs caught by single-race debugging before trusting any full run (sign inversion between faster-positive offsets and slower-positive lap-time terms; off-by-one giving the winner's pace to a field car). Formal re-run, all races reported (n=65; three more 2025 rounds ingested since the original 62):
- strategy match 63/65 = 96.9 pct (was 3/62 = 4.8)
- winner in P10-P90 band 23/65 = 35.4 pct (was 0/62 = 0)

CAVEATS (stated on purpose): strategy-match is structurally easier with pace-aware sims - it now measures whether the winning plan was near-optimal GIVEN pre-race pace knowledge. Band coverage 35 pct beats 0 but sits below nominal 80 pct (intra-season development, qualifying-specific pace, SC luck unmodeled). Next lever for both D22's tail gap and band coverage: sharper pace estimates (qualifying-based) and a car-development trend term.
