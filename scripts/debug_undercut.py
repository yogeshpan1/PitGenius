import sys
sys.path.insert(0, ".")
import pandas as pd
from scripts.backtest_undercut import attempt_features
from pitgenius.data import store
from pitgenius.analysis import undercut_history
from pitgenius.models.degradation_model import TireDegradationModel

conn = store.connect()
laps = store.load_laps(conn=conn)
pits = store.load_pit_stops(conn=conn)
conn.close()

att = pd.read_parquet("reports/_attempts_cache.parquet")
feats = attempt_features(laps, att)
cols = ["year", "round", "kind", "gap_at_attempt_s", "signed_gap_s",
        "attacker_compound", "attacker_age", "defender_compound",
        "defender_age", "success"]
print(feats[cols].head(8).to_string())
print()
print("age stats:", feats[["attacker_age", "defender_age"]].describe().to_string())

# Direct probe: same scenario, two gaps
m = TireDegradationModel().load()
rows = pd.DataFrame({
    "tire_life": [2, 15], "fuel_proxy": [0.0, 0.0],
    "tire_compound": ["MEDIUM", "MEDIUM"], "circuit": ["1_2022", "1_2022"]})
p = m.predict(rows)
print("fresh(2):", p.p50[0], " old(15):", p.p50[1])