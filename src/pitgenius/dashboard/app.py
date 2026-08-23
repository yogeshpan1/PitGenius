"""PitGenius historical race explorer — Streamlit dashboard.

Run:  streamlit run src/pitgenius/dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pitgenius.analysis import degradation, pitstops, undercut_history
from pitgenius.config import format_lap_time
from pitgenius.data import store

st.set_page_config(page_title="PitGenius", page_icon="🏁", layout="wide")

COMPOUND_COLORS = {
    "SOFT": "#ff4b4b", "MEDIUM": "#ffd23f", "HARD": "#e8e8e8",
    "INTERMEDIATE": "#3fb950", "WET": "#2f6fed", "UNKNOWN": "#888888",
}


@st.cache_data(ttl=600)
def load_all():
    conn = store.connect()
    try:
        races = store.load_races(conn)
        laps = store.load_laps(conn)
        pits = store.load_pit_stops(conn)
        results = store.load_results(conn)
    finally:
        conn.close()
    return races, laps, pits, results


races, laps, pits, results = load_all()

if races.empty:
    st.error(
        "No data ingested yet. Run: "
        "`python -m src.pitgenius.data.ingest --seasons 2021 2022 2023 2024 2025`"
    )
    st.stop()

# ---------------------------------------------------------------- sidebar --
st.sidebar.title("🏁 PitGenius")
years = sorted(races["year"].unique(), reverse=True)
year = st.sidebar.selectbox("Season", years)
season_races = races[races["year"] == year].sort_values("round")
race_label = {
    r["round"]: f"R{r['round']} — {r['event_name']}"
    for _, r in season_races.iterrows()
}
round_ = st.sidebar.selectbox(
    "Race",
    list(race_label.keys()),
    format_func=lambda r: race_label.get(r, f"R{r}"),
)

sel_laps = laps[(laps["year"] == year) & (laps["round"] == round_)]
sel_pits = pits[(pits["year"] == year) & (pits["round"] == round_)]
sel_results = results[(results["year"] == year) & (results["round"] == round_)]

tab_race, tab_tires, tab_pits, tab_undercut = st.tabs(
    ["Race overview", "Tire degradation", "Pit stops", "Undercut / overcut"]
)

# ----------------------------------------------------------- race overview --
with tab_race:
    st.header(f"{year} R{round_} — "
              f"{races.loc[(races['year'] == year) & (races['round'] == round_), 'event_name'].iloc[0]}")

    if not sel_results.empty:
        podium = sel_results.dropna(subset=["position"]).nsmallest(3, "position")
        cols = st.columns(3)
        for c, (_, row) in zip(cols, podium.iterrows()):
            c.metric(f"P{int(row['position'])} {row['driver']}",
                     row.get("team", ""),
                     f"{row['points']:.0f} pts" if pd.notna(row["points"]) else "")

        finished = sel_results.dropna(subset=["position"]).sort_values("position")
        dnf = sel_results[sel_results["position"].isna()]
        st.subheader("Classification")
        show = finished[["position", "driver", "team", "grid_position",
                         "status", "points"]].reset_index(drop=True)
        if not dnf.empty:
            show = pd.concat([show, dnf[["position", "driver", "team",
                                         "grid_position", "status", "points"]]
                              .assign(position="DNF")], ignore_index=True)
        st.dataframe(show, use_container_width=True, height=500)

    # Position-vs-lap trace for top finishers
    if not sel_laps.empty:
        st.subheader("Position by lap (top 10 finishers)")
        top10 = finished.head(10)["driver"].tolist() if not sel_results.empty else []
        pl = sel_laps[sel_laps["driver"].isin(top10) & sel_laps["position"].notna()]
        fig = px.line(pl, x="lap_number", y="position", color="driver",
                      category_orders={"driver": top10}, yaxis=dict(autorange="reversed"))
        fig.update_layout(height=420, yaxis_title="Position", xaxis_title="Lap")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------- tire degradation --
with tab_tires:
    st.header("Tire degradation — lap-time delta vs tire age")
    compound = st.selectbox("Compound", ["MEDIUM", "SOFT", "HARD"])
    curve = degradation.degradation_curve(laps, (year, round_), compound)
    if curve.empty or len(curve) < 4:
        st.info("Not enough clean green-flag data for this compound/race.")
    else:
        slope, intercept, r2 = degradation.fit_linear_degradation(curve)
        c1, c2, c3 = st.columns(3)
        c1.metric("Degradation rate", f"{slope:+.3f} s/lap"
                  if np.isfinite(slope) else "—")
        c2.metric("R² (linear fit)", f"{r2:.2f}" if np.isfinite(r2) else "—")
        c3.metric("Clean-lap sample", f"{int(curve['n_obs'].sum())} laps")

        fig = go.Figure()
        fig.add_scatter(x=curve["tire_life"], y=curve["mean_delta"],
                        mode="lines+markers", name=f"{compound} mean delta",
                        line=dict(color=COMPOUNDS_COLORS.get(compound, "#888")))
        xs = np.linspace(curve["tire_life"].min(), curve["tire_life"].max(), 50)
        if np.isfinite(slope):
            fig.add_scatter(x=xs, y=slope * xs + intercept, mode="lines",
                            name="linear fit", line=dict(dash="dash"))
        fig.update_layout(xaxis_title="Tire age (laps)",
                          yaxis_title="Δ vs driver median clean lap (s)",
                          height=420)
        st.plotly_chart(fig, use_container_width=True)

    # All-compound stint scatter for the selected race
    gf = degradation.green_flag_laps(sel_laps)
    if not gf.empty:
        med = gf.groupby("driver")["lap_time_s"].transform("median")
        gf = gf.assign(delta=gf["lap_time_s"] - med)
        fig2 = px.scatter(gf, x="tire_life", y="delta", color="tire_compound",
                          color_discrete_map=COMPOUND_COLORS,
                          labels={"tire_life": "Tire age (laps)",
                                  "delta": "Δ vs median (s)"},
                          opacity=0.55, hover_data=["driver"])
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------------------- pit stops --
with tab_pits:
    st.header("Pit stops")
    if sel_pits.empty:
        st.info("No pit stops recorded for this race.")
    else:
        tl = pitstops.pit_timeline(pits, year, round_)
        fig = go.Figure()
        for driver, g in tl.groupby("driver"):
            fig.add_scatter(x=g["lap"], y=[driver] * len(g), mode="markers",
                            marker=dict(symbol="line-ns-open", size=14,
                                        line=dict(width=3)),
                            name=driver, showlegend=False,
                            hovertext=[
                                f"{driver} stop {i+1}: L{l}"
                                + (f" ({d:.1f}s)" if pd.notna(d) else "")
                                for i, (l, d) in enumerate(zip(g["lap"], g["duration_s"]))
                            ])
        fig.update_layout(height=max(360, 22 * tl["driver"].nunique()),
                          xaxis_title="Lap", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        stats = pitstops.pit_loss_stats(laps, pits)
        per_race_loss = stats.get("per_race", {}).get((year, round_))
        c1, c2, c3 = st.columns(3)
        c1.metric("Stops recorded", len(tl))
        c2.metric("Est. pit loss this race",
                  f"{per_race_loss:.1f}s" if per_race_loss is not None and np.isfinite(per_race_loss) else "—")
        c3.metric("Global median pit loss",
                  f"{stats.get('median_pit_loss_s', float('nan')):.1f}s"
                  if stats.get("n") else "—")
        st.caption("Pit loss estimated from out-lap vs own median clean lap; "
                   "SC-contaminated stops are filtered but some noise remains.")

# ------------------------------------------------------- undercut / overcut --
with tab_undercut:
    st.header("Undercut / overcut outcomes")
    @st.cache_data(ttl=3600, show_spinner="Detecting undercut/overcut attempts across all races…")
    def all_attempts():
        return undercut_history.find_attempts(laps, pits)

    att = all_attempts()
    if att.empty:
        st.info("No adjacent-car stop sequences detected yet.")
    else:
        summary = undercut_history.summarize_attempts(att)
        st.dataframe(summary, use_container_width=True)
        st.caption("Success = attacker ahead of the rival once both have pitted. "
                   "Counts shown alongside rates (no cherry-picking).")

        sel = att[(att["year"] == year) & (att["round"] == round_)]
        st.subheader(f"This race ({len(sel)} attempts)")
        if not sel.empty:
            st.dataframe(sel, use_container_width=True)

        fig = px.histogram(att, x="gap_at_attempt_s", color="kind", nbins=40,
                           barmode="overlay", opacity=0.6,
                           labels={"gap_at_attempt_s": "Gap at attempt (s)"})
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.caption(f"Laps loaded: {len(laps):,} across {len(races)} races")