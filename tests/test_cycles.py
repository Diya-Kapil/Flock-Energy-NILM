"""Cycle detection on hand-built signals with known answers."""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.cycles import CycleParams, detect_cycles, on_runs


def signal(*parts) -> np.ndarray:
    """Concatenate (value, minutes) blocks into a 1-minute power series."""
    return np.concatenate([np.full(minutes, float(value)) for value, minutes in parts])


class TestDetection:
    def test_known_cycle(self):
        cycles = detect_cycles(signal((0, 30), (2000, 30), (0, 30)))
        assert len(cycles) == 1
        row = cycles.iloc[0]
        assert (row["start_pos"], row["end_pos"], row["duration_min"]) == (30, 59, 30)
        assert row["peak_w"] == 2000
        assert row["energy_kwh"] == pytest.approx(1.0)
        assert row["heat_minutes"] == 30
        assert row["heated"]
        assert row["complete"]

    def test_short_burst_is_rejected(self):
        assert detect_cycles(signal((0, 30), (2000, 19), (0, 30))).empty

    def test_exactly_minimum_duration_is_kept(self):
        assert len(detect_cycles(signal((0, 30), (2000, 20), (0, 30)))) == 1

    def test_low_peak_is_rejected(self):
        assert detect_cycles(signal((0, 30), (150, 40), (0, 30))).empty

    def test_unheated_cycle_is_kept_and_marked(self):
        row = detect_cycles(signal((0, 30), (300, 40), (0, 30))).iloc[0]
        assert not row["heated"]
        assert row["heat_energy_kwh"] == 0

    def test_standby_is_not_operation(self):
        assert detect_cycles(signal((5, 200))).empty
        assert detect_cycles(signal((20, 200))).empty  # Exactly at the threshold is not "above".

    def test_gap_at_tolerance_is_bridged(self):
        cycles = detect_cycles(signal((0, 30), (2000, 15), (0, 10), (300, 15), (0, 30)))
        assert len(cycles) == 1
        assert cycles.iloc[0]["duration_min"] == 40
        assert cycles.iloc[0]["on_minutes"] == 30

    def test_gap_above_tolerance_splits(self):
        assert len(detect_cycles(signal((0, 30), (2000, 25), (0, 11), (300, 25), (0, 30)))) == 2

    def test_missing_minutes_are_not_bridged(self):
        power = signal((0, 30), (2000, 25), (0, 3), (300, 25), (0, 30))
        power[56] = np.nan
        cycles = detect_cycles(power)
        assert len(cycles) == 2
        assert not cycles["complete"].any()

    def test_missing_minute_beside_a_cycle_marks_it_incomplete(self):
        power = signal((0, 30), (2000, 25), (0, 30))
        power[60] = np.nan
        assert not detect_cycles(power).iloc[0]["complete"]

    def test_missing_minute_far_away_leaves_it_complete(self):
        power = signal((0, 30), (2000, 25), (0, 30))
        power[80] = np.nan
        assert detect_cycles(power).iloc[0]["complete"]

    def test_energy_ignores_missing_minutes_inside_nothing(self):
        assert detect_cycles(np.full(100, np.nan)).empty

    def test_custom_parameters_are_used(self):
        params = CycleParams(on_w=100, max_gap_min=2, min_duration_min=5, min_peak_w=100)
        assert len(detect_cycles(signal((0, 5), (200, 5), (0, 5)), params)) == 1


def test_on_runs_empty():
    starts, ends = on_runs(np.zeros(10), 20, 10)
    assert len(starts) == 0 and len(ends) == 0
