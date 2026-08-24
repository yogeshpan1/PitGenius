"""Consolidated unit tests: scoring metrics, publisher immutability,
undercut calculator math, tire cliff detection, sim-racing parsers,
safety-car counting. All pure-function tests — no network, no FastF1.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pitgenius.backtest.scoring import (
    interval_coverage, pinball_loss, quantile_scorecard)
from pitgenius.models.degradation_model import DegradationPrediction
from pitgenius.live import publisher
from pitgenius.models.safety_car import count_interruptions
from pitgenius.models.tire_cliff import detect_cliff
from pitgenius.simracing.bridge import parse_acc_car_update, pit_advice
from pitgenius.strategy import undercut_calc


# ------------------------------------------------------------- scoring ----
def test_pinball_loss_median_is_half_mae():
    y = np.array([1.0, 2.0, 3.0])
    assert pinball_loss(y, y, 0.5) == pytest.approx(0.0)
    pred = np.array([2.0, 2.0, 2.0])
    assert pinball_loss(y, pred, 0.5) == pytest.approx(
        np.abs(y - pred).mean() / 2)


def test_coverage_perfect_and_zero():
    y = np.zeros(10)
    assert interval_coverage(y, -1, 1) == 1.0
    assert interval_coverage(y, 1, 2) == 0.0


def test_scorecard_keys():
    sc = quantile_scorecard(np.array([1.0, 2.0]),
                            np.array([0.5, 1.5]), np.array([1.0, 2.0]),
                            np.array([1.5, 2.5]))
    assert sc["n"] == 2 and sc["coverage_80"] == 1.0


# ---------------------------------------------------------- publisher -----
@pytest.fixture()
def tmp_pred_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(publisher, "PREDICTIONS_DIR", tmp_path / "preds")
    return tmp_path


def test_publish_is_immutable(tmp_pred_dir):
    publisher.publish_pre_race(2026, 1, "Test GP",
                               {"predicted_winner": {"driver": "VER"}})
    with pytest.raises(FileExistsError):
        publisher.publish_pre_race(2026, 1, "Test GP", {"x": 1})


def test_tamper_detection(tmp_pred_dir):
    publisher.publish_pre_race(2026, 2, "Test GP", {"a": 1})
    path = (tmp_pred_dir / "preds" / "2026" / "r2" / "pre_race.json")
    payload = json.loads(path.read_text())
    payload["predictions"]["a"] = 999   # tamper
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        publisher.load_published(2026, 2)


def test_score_roundtrip(tmp_pred_dir):
    publisher.publish_pre_race(
        2026, 3, "Test GP",
        {"predicted_winner": {"driver": "VER"},
         "podium_probabilities": {"drivers": ["VER", "NOR", "PIA"]}})
    out = publisher.score_post_race(
        2026, 3, {"winner": "NOR", "podium": ["NOR", "PIA", "LEC"]})
    score = json.loads(out.read_text())
    assert score["checks"]["winner_correct"] is False
    assert score["checks"]["podium_overlap"] == 2


# ------------------------------------------------------------ undercut ----
class FakeModel:
    """Returns fixed quantiles: fresh tires ~0, old tires degrade linearly."""

    def predict(self, rows):
        ages = rows["tire_life"].to_numpy()
        p50 = np.clip((ages - 3) * 0.15, -0.05, None)
        return DegradationPrediction(
            p10=p50 - 0.2, p50=p50, p90=p50 + 0.2)


def test_undercut_favored_when_gap_small():
    call = undercut_calc.evaluate_move(
        FakeModel(), kind="undercut", gap_s=1.0,
        attacker_compound="MEDIUM", attacker_age=12,
        defender_compound="MEDIUM", defender_age=13, circuit="t")
    assert call.p_flip > 0.8


def test_undercut_unfavored_when_gap_large():
    call = undercut_calc.evaluate_move(
        FakeModel(), kind="undercut", gap_s=25.0,
        attacker_compound="MEDIUM", attacker_age=12,
        defender_compound="MEDIUM", defender_age=13, circuit="t")
    assert call.p_flip < 0.2


# ------------------------------------------------- adjacent-stop pairs -----
def _mini_race():
    """Synthetic 4-car race: pace order A>B>C>D, stops at laps 8/9/10/14."""
    rows = []
    pace = {"A": 80.0, "B": 80.5, "C": 81.0, "D": 81.5}
    for d, t in pace.items():
        for lap in range(1, 21):
            rows.append({"year": 2024, "round": 1, "driver": d,
                         "lap_number": lap, "lap_time_s": t})
    return pd.DataFrame(rows)


def test_find_adjacent_pairs_windows_and_labels():
    from pitgenius.analysis.undercut_history import find_adjacent_pairs
    laps = _mini_race()
    pits = pd.DataFrame({
        "year": [2024] * 4, "round": [1] * 4,
        "driver": ["C", "D", "A", "B"],
        "lap": [8, 9, 10, 14],
    })
    pairs = find_adjacent_pairs(laps, pits)
    # Exactly two canonical events: C->D (1 lap apart) and A->B (4 apart).
    assert len(pairs) == 2
    cd = pairs[(pairs["attacker"] == "C") & (pairs["defender"] == "D")]
    ab = pairs[(pairs["attacker"] == "A") & (pairs["defender"] == "B")]
    assert len(cd) == 1 and len(ab) == 1
    # Responder behind -> 'undercut'; success = first pitter retains lead.
    assert bool(cd.iloc[0]["is_attempt"]) is True
    assert cd.iloc[0]["kind"] == "undercut" and bool(cd.iloc[0]["success"])
    assert bool(ab.iloc[0]["is_attempt"]) is False
    assert ab.iloc[0]["kind"] == "undercut" and bool(ab.iloc[0]["success"])
    # No double-counting: the reversed orientation must not exist.
    assert not ((pairs["attacker"] == "D") &
                (pairs["defender"] == "C")).any()



def test_cliff_detected_on_synthetic_stint():
    rng = np.random.default_rng(0)
    age = np.arange(1, 26)
    delta = np.where(age <= 15, age * 0.02, 0.30 + (age - 15) * 0.35)
    df = pd.DataFrame({"tire_life": age, "delta": delta + rng.normal(0, .01, 25)})
    r = detect_cliff(df)
    assert r["cliff_lap"] == 15


def test_no_cliff_on_linear_stint():
    age = np.arange(1, 21)
    df = pd.DataFrame({"tire_life": age, "delta": age * 0.05})
    assert detect_cliff(df)["cliff_lap"] is None


# --------------------------------------------------------- simracing ------
ACC_CAR_UPDATE_FMT = struct.Struct("<H 3f I f f f B B")


def _acc_packet(speed=280.0, lap_ms=91_234):
    buf = bytearray(64)
    ACC_CAR_UPDATE_FMT.pack_into(
        buf, 0, 3, 100.0, 200.0, 300.0, lap_ms, 0.5, 1234.5, 11, 200, 7)
    struct.pack_into("<f", buf, 60, speed)
    return bytes(buf)


def test_acc_parse_roundtrip():
    st = parse_acc_car_update(_acc_packet())
    assert st.car_index == 3
    assert st.lap_time_ms == 91_234
    assert st.speed_kmh == pytest.approx(280.0)


def test_acc_parse_rejects_short():
    with pytest.raises(ValueError):
        parse_acc_car_update(b"\x00\x01")


def test_pit_advice_states():
    from pitgenius.simracing.bridge import CarState
    s = CarState(0, 250.0, 4, 11000, 90000)
    assert pit_advice(s, planned_stop_lap=20, current_lap=20) == "BOX NOW"
    assert pit_advice(s, 22, 21) == "box in 1"
    assert pit_advice(s, 30, 12) == "stay out"


# -------------------------------------------------------- safety car ------
def test_count_interruptions_transitions():
    laps = pd.DataFrame({
        "year": [2024] * 10, "round": [1] * 10,
        "lap_number": range(1, 11),
        "track_status": ["1", "1", "4", "4", "1", "1", "67", "67", "1", "1"],
    })
    c = count_interruptions(laps)
    assert c.iloc[0]["n_periods"] == 2
    assert c.iloc[0]["n_interrupted_laps"] == 4