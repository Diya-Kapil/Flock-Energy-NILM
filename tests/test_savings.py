"""Opportunity sizing and sample-size arithmetic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import savings


def cycles_with_heat(heat) -> pd.DataFrame:
    return pd.DataFrame({"heat_energy_kwh": heat, "complete": True, "heated": True})


class TestOwnCycleScenarios:
    def test_known_answer(self):
        row = savings.own_cycle_scenarios(cycles_with_heat([0.4, 0.5, 0.6, 1.0, 1.5]), valid_days=365, quantile=0.5)
        assert row["cap_heat_kwh_per_cycle"] == 0.6
        assert row["saving_kwh_per_year"] == pytest.approx(0.4 + 0.9)
        assert row["cycles_reduced_pct"] == 40

    def test_annualises_for_days_observed(self):
        row = savings.own_cycle_scenarios(cycles_with_heat([0.5, 0.5, 1.5]), valid_days=73, quantile=0.5)
        assert row["saving_kwh_per_year"] == pytest.approx(1.0 * 5)

    def test_unheated_and_incomplete_cycles_set_no_cap(self):
        frame = pd.DataFrame({"heat_energy_kwh": [0.5, 0.5, 9.0, 0.0], "complete": [True, True, False, True], "heated": [True, True, True, False]})
        assert savings.own_cycle_scenarios(frame, 365, 0.5)["cap_heat_kwh_per_cycle"] == 0.5


def test_peak_figures_known_answer():
    minutes = pd.date_range("2014-05-07 15:00", periods=120, freq="min")
    table = pd.DataFrame({"uk_local_minute": minutes, "valid": True, "washing_machine_w": np.r_[np.zeros(60), np.full(60, 600.0)],
                          "aggregate_w": 1000.0})
    row = savings.peak_figures(table)
    assert row["machine_energy_in_peak_pct"] == 100
    assert row["mean_machine_w_in_peak"] == 600
    assert row["machine_share_of_peak_aggregate_pct"] == 60


class TestSampleSize:
    def test_known_answer(self):
        assert savings.sample_size_per_arm(0.4, 0.1) == int(np.ceil(2 * savings.Z_ALPHA_POWER_SQUARED * 16))

    def test_smaller_effect_needs_more_households(self):
        assert savings.sample_size_per_arm(0.4, 0.01) > savings.sample_size_per_arm(0.4, 0.1)

    def test_adjustment_reduces_the_requirement(self):
        assert savings.sample_size_per_arm(0.4, 0.05, 0.5) < savings.sample_size_per_arm(0.4, 0.05)

    def test_zero_effect_is_rejected(self):
        with pytest.raises(ValueError):
            savings.sample_size_per_arm(0.4, 0)


def test_measurement_power_favours_the_sub_meter():
    opportunity = pd.DataFrame({"machine_kwh_per_year": [100.0, 200.0, 300.0], "household_kwh_per_year": [3000.0, 4000.0, 5000.0],
                                "saving_cap_at_median_kwh_per_year": [20.0, 40.0, 60.0]})
    table = savings.measurement_power(opportunity).set_index("measured_at")
    assert table.loc["Washing machine sub-meter", "households_per_arm_unadjusted"] < table.loc["Whole-house smart meter", "households_per_arm_unadjusted"]
