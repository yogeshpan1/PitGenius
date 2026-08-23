import sys
sys.path.insert(0, ".")
from pitgenius.data import store
from pitgenius.analysis import undercut_history

conn = store.connect()
laps = store.load_laps(conn=conn)
pits = store.load_pit_stops(conn=conn)
att = undercut_history.find_attempts(laps, pits)
a = att.iloc[0]
lap_idx = laps.set_index(["year", "round", "driver", "lap_number"])
key = (a["year"], a["round"], a["attacker"], a["attacker_pit_lap"])
out = [
    "att dtypes: " + str(att[["year", "round", "attacker_pit_lap"]].dtypes.to_dict()),
    "laps dtypes: " + str(laps[["year", "round", "driver", "lap_number"]].dtypes.to_dict()),
    "key: " + repr(key) + " types: " + str([type(k).__name__ for k in key]),
]
try:
    lap_idx.loc[key]
    out.append("LOOKUP OK")
except KeyError as e:
    out.append("KeyError: " + repr(e))
    sub = laps[(laps["year"] == a["year"]) & (laps["round"] == a["round"])
               & (laps["driver"] == a["attacker"])]
    near = sub[sub["lap_number"].between(a["attacker_pit_lap"] - 1,
                                         a["attacker_pit_lap"] + 1)]
    out.append("nearby rows: "
               + str(near[["lap_number", "tire_compound", "tire_life"]]
                     .to_dict(orient="records")))
open("data/dbg.txt", "w").write(chr(10).join(out))
print("written")