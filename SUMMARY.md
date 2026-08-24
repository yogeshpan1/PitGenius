# PitGenius — Build Summary

## What was built and validated

| Component | Status | Validation |
|---|---|---|
| FastF1 ingestion → SQLite (2021–2025 races, resumable, rate-limit aware) | ✅ Built, run | Manifest-tracked; failures logged not hidden |
| Streamlit historical explorer (degradation curves, pit timelines, undercut outcomes) | ✅ Built | Visual inspection |
| Tire degradation quantile model (LightGBM P10/P50/P90) | ✅ Built + trained | LORO over 102 races: MAE 0.70s, coverage 61.7%; 2025 holdout: MAE 0.95s, coverage 68.4% (below 80% target — documented, not fudged) |
| Pre-race car/team pace model (rolling offsets) | ✅ Built + validated in isolation | 98% coverage; Spearman vs finish +0.594; reproduces known eras (RB rank-1 in 2023, McLaren top 2024–25, Haas −1.47 in 2021) — `scripts/validate_pace_model.py` |
| Undercut/overcut calculator | ⚠️ Beats constant guessing; pace features did NOT close the kind-prior gap | D22 fixes 1–5 + D23 part 2: Brier 0.1317 with pace features vs 0.1295 without (constant 0.1373 ✅, non-leaky kind-prior 0.1270 ❌). Details: `reports/undercut_backtest.json` + DECISIONS.md D22/D23 |
| Monte Carlo simulator + SC incidence model | ✅ Pace-aware; formal backtest re-run | Strategy match **63/65 = 96.9%** (was 3/62); winner in P10–P90 band **23/65 = 35.4%** (was 0/62). Caveats in DECISIONS.md D23: match metric now pace-conditioned; band still below nominal 80% |
| Formal backtesting framework | ✅ Built + run | 62 races, temporal protocol: recommended strategy matched winner's actual strategy 3/62 (4.8%); winner inside sim's P10–P90 band 0/62 — the simulator does not model car dominance. All races reported |
| Live prediction infra (immutable hash-sealed publish/score/social) | ✅ Built | 3 unit tests incl. tamper detection; **awaiting first real race weekend** |
| LLM explainability w/ confidence gating | ✅ Built, **stubbed** | Needs `LLM_API_KEY` |
| Tire cliff detection | ✅ Built | Synthetic-stint unit tests (exact breakpoint recovery) |
| Sim-racing bridge (ACC UDP parse, iRacing adapter, pit advice) | ✅ Built | Parser unit tests; live test pending (needs sim running) |
| FastAPI backend + React frontend | ✅ Built | Manual run: `uvicorn src.pitgenius.api.main:app`; `cd frontend && npm i && npm run dev` |
| Test suite | ✅ 15 passing | `pytest tests/ -q` |

## Stubbed — what YOU need to supply

1. **LLM key** — set env vars before using explanations:
   ```
   set LLM_API_KEY=gsk_...        (Groq)
   set LLM_BASE_URL=https://api.groq.com/openai/v1   (optional)
   set LLM_MODEL=openai/gpt-oss-120b                (optional; llama-3.3-70b-versatile is deprecated on Groq)
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