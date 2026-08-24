# PitGenius — Build Summary

## What was built and validated

| Component | Status | Validation |
|---|---|---|
| FastF1 ingestion → SQLite (2021–2025 races, resumable, rate-limit aware) | ✅ Built, run | Manifest-tracked; failures logged not hidden |
| Streamlit historical explorer (degradation curves, pit timelines, undercut outcomes) | ✅ Built | Visual inspection |
| Tire degradation quantile model (LightGBM P10/P50/P90) | ✅ Built + trained | LORO over 102 races: MAE 0.70s, coverage 61.7%; 2025 holdout: MAE 0.95s, coverage 68.4% (below 80% target — documented, not fudged) |
| Undercut/overcut calculator | ⚠️ Built + backtested — **STILL FAILING after D22 fixes** | All three D22 fixes implemented + re-run: all-pairs eval n=1852 → Brier 0.5005 vs baseline 0.1383; P(attempt) AUC 0.532 (no signal); fresh-tire delta fixed −1.76s→+0.75s. Details in `reports/undercut_backtest.json` + DECISIONS.md D22 update. Do not trust its probabilities |
| Monte Carlo simulator + SC incidence model | ✅ Built | Unit tests; full-stack retrospective backtest (`reports/backtest_report.md`) |
| Formal backtesting framework | ✅ Built + run | 62 races, temporal protocol: recommended strategy matched winner's actual strategy 3/62 (4.8%); winner inside sim's P10–P90 band 0/62 — the simulator does not model car dominance. All races reported |
| Live prediction infra (immutable hash-sealed publish/score/social) | ✅ Built | 3 unit tests incl. tamper detection; **awaiting first real race weekend** |
| LLM explainability w/ confidence gating | ✅ Built, **stubbed** | Needs `LLM_API_KEY` |
| Tire cliff detection | ✅ Built | Synthetic-stint unit tests (exact breakpoint recovery) |
| Sim-racing bridge (ACC UDP parse, iRacing adapter, pit advice) | ✅ Built | Parser unit tests; live test pending (needs sim running) |
| FastAPI backend + React frontend | ✅ Built | Manual run: `uvicorn src.pitgenius.api.main:app`; `cd frontend && npm i && npm run dev` |
| Test suite | ✅ 14 passing | `pytest tests/ -q` |

## Stubbed — what YOU need to supply

1. **LLM key** — set env vars before using explanations:
   ```
   set LLM_API_KEY=gsk_...        (Groq)
   set LLM_BASE_URL=https://api.groq.com/openai/v1   (optional)
   set LLM_MODEL=llama-3.3-70b-versatile             (optional)
   ```
   Without it, `explain()` returns a marked `[STUB]` template.
2. **AWS credentials** — follow `docs/deployment.md`. Config is written;
   nothing was deployed (Academy sandbox can't hold resources).
3. **Social posting** — each scored race writes
   `predictions/<season>/r<round>/social_post.txt` for you to review and
   post manually. To automate: add an X/LinkedIn API call in
   `publisher.social_post`.
4. **ACC/iRacing live telemetry** — parsers are unit-tested against
   synthetic packets only. Run the end-to-end check in `docs/simracing.md`
   with the sim actually running.

## Structurally impossible to finish today

- **The Stage 7 live loop cannot produce real predictions until a real
  session exists.** The infrastructure is complete and runnable; it needs a
  race weekend.

## How to run the live loop going forward

Before each remaining race weekend this season:

```bash
# Thursday/Friday (before the race):
python scripts/live_race.py publish --season <year> --round <n> --event "<GP name>"

# Sunday evening (after the race):
python scripts/live_race.py score --season <year> --round <n> --winner <CODE> --podium <P1>,<P2>,<P3> --strategy "<winner's strategy>"
python scripts/live_race.py social --season <year> --round <n> --event "<GP name>"
git add predictions && git commit -m "track record: <GP name>" && git push
```

Predictions are immutable once published (hash-sealed); scoring verifies
the seal. Over a season this builds the public, honestly-graded track
record that is the whole point of the project.

## Headline honesty statement

The degradation model interpolates well (MAE ~0.7s) but under-covers its
intervals. The undercut calculator **failed** its backtest. The full-stack
simulator's strategy recommendation matched the actual winner only 3/62
times, and its uncertainty bands are too wide to be informative about a
dominant car. These negative results are published deliberately: they are
the project's real current state, and each has a diagnosed cause and a
planned fix (DECISIONS.md D20/D22).

## Known limitations (also in docs/methodology.md §7)

Circuit identity keyed by round number · linear fuel proxy · no weather
features · cumulative-time running-order proxies misorder after SC restarts
· small per-circuit samples. Read §7 before quoting any number publicly.