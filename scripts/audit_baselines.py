"""D22 fix-4/5 audits.

Part A: orientation audit of the pre-fix report (why p_flip was
anti-correlated: it returned P(first pitter LOSES) against retention labels).

Part B: fair (non-leaky) baseline comparison for the current calibrated
model. The kind-baseline in the report is fitted on TEST-season outcomes
itself (an oracle); the non-leaky version uses per-kind rates from all
OTHER seasons, which is what any real prior-knowledge predictor could do.
"""
import json

import numpy as np
import pandas as pd

rep = json.load(open("reports/undercut_backtest.json", encoding="utf-8"))
res = pd.DataFrame(rep["per_pair"])

print("== Part B: fair (non-leaky) baselines vs model ==")
rows = []
for year, g in res.groupby("year"):
    y = g["success"].to_numpy()
    r = y.mean()
    other = res[res["year"] != year]
    prates = other.groupby("kind")["success"].mean()
    mapped = g["kind"].map(prates).fillna(other["success"].mean())
    rows.append({
        "year": int(year), "n": len(g),
        "model": np.mean((g["p_predicted"] - y) ** 2),
        "constant_base_rate": np.mean((r - y) ** 2),
        "kind_leaky_oracle": np.mean(
            (g["kind"].map(g.groupby("kind")["success"].mean()) - y) ** 2),
        "kind_nonleaky_prior_years": np.mean((mapped - y) ** 2),
    })
cmp = pd.DataFrame(rows).set_index("year")
print(cmp.round(4).to_string())
w = cmp["n"] / cmp["n"].sum()
print("\nPooled (n-weighted):")
for col in cmp.columns:
    if col == "n":
        continue
    print(f"  {col:<28} {np.average(cmp[col], weights=w):.4f}")
