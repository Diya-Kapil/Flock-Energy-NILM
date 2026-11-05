"""Leakage tests: features use aggregate power only, and look backwards only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.dataset import build_house, downsample
from src.models.features import feature_names, make_features

FORBIDDEN = ("appliance", "washing", "machine", "issues", "house", "hour", "weekday", "month")


def series(n: int = 400, seed: int = 3) -> np.ndarray:
    return np.abs(np.random.default_rng(seed).normal(500, 300, n))


class TestNoFutureInformation:
    @pytest.mark.parametrize("step", [1, 15, 30])
    def test_changing_the_future_never_changes_earlier_features(self, step):
        a = series()
        changed = a.copy()
        cut = 200
        changed[cut + 1:] = changed[cut + 1:] * 7 + 1000
        first, second = make_features(a, step), make_features(changed, step)
        pd.testing.assert_frame_equal(first.iloc[: cut + 1], second.iloc[: cut + 1])

    def test_primary_features_have_no_future_columns(self):
        assert not [name for name in feature_names() if "future" in name]

    def test_retrospective_features_are_labelled_and_do_use_the_future(self):
        a = series()
        changed = a.copy()
        changed[201:] += 1000
        first, second = make_features(a, retrospective=True), make_features(changed, retrospective=True)
        differ = [c for c in first.columns if not np.allclose(first[c].iloc[:200], second[c].iloc[:200], equal_nan=True)]
        assert differ
        assert all(name.startswith("future_") for name in differ)


class TestAggregateOnly:
    def test_feature_names_never_name_other_inputs(self):
        for name in feature_names() + feature_names(30):
            assert not any(word in name for word in FORBIDDEN), name

    def _table(self) -> pd.DataFrame:
        n = 300
        a = series(n)
        return pd.DataFrame({"aggregate_w": a.astype("float32"), "washing_machine_w": np.zeros(n, dtype="float32"), "issues_any": False,
                             "other_appliance1_w": np.zeros(n, dtype="float32"), "valid": True, "missing": False})

    def test_changing_appliance_issues_or_house_never_changes_features(self):
        table = self._table()
        changed = table.copy()
        changed["washing_machine_w"] = 2000.0
        changed["other_appliance1_w"] = 999.0
        changed["issues_any"] = True
        first, second = build_house(table, 1), build_house(changed, 16)
        pd.testing.assert_frame_equal(first.X, second.X)

    def test_target_is_the_washing_machine(self):
        table = self._table()
        table["washing_machine_w"] = 123.0
        assert (build_house(table, 1).y == 123).all()


class TestKnownValues:
    def test_lag_change_and_rolling_values(self):
        a = np.arange(10, dtype="float64") * 100
        features = make_features(a)
        assert features["lag_1m"].iloc[5] == 400
        assert features["change_2m"].iloc[5] == 200
        assert features["mean_5m"].iloc[9] == pytest.approx(700)
        assert features["min_5m"].iloc[9] == 500
        assert features["max_5m"].iloc[9] == 900

    def test_counts_large_rises_and_time_since_heater(self):
        a = np.r_[np.full(10, 200.0), np.full(10, 2300.0)]
        features = make_features(a)
        assert features["large_rises_60m"].iloc[15] == 1
        assert features["minutes_since_heater_rise"].iloc[15] == 5
        assert features["minutes_since_heater_rise"].iloc[5] == 240

    def test_missing_input_stays_missing(self):
        a = series(50)
        a[20] = np.nan
        features = make_features(a)
        assert np.isnan(features["aggregate"].iloc[20])
        assert np.isnan(features["lag_1m"].iloc[21])
        assert np.isnan(features["change_1m"].iloc[21])

    def test_windows_scale_with_interval(self):
        names = feature_names(30)
        assert "mean_60m" in names
        assert "mean_5m" not in names
        assert "lag_1m" not in names


class TestFrozenExclusion:
    def test_frozen_values_cannot_enter_neighbouring_windows(self):
        n = 300
        a = series(n)
        frozen = np.zeros(n, dtype=bool)
        frozen[100:170] = True
        table = pd.DataFrame({"aggregate_w": a.astype("float32"), "washing_machine_w": np.zeros(n, dtype="float32"),
                              "valid": ~frozen, "missing": False})
        changed = table.copy()
        changed.loc[frozen, "aggregate_w"] = 2169.0
        first, second = build_house(table, 1), build_house(changed, 1)
        pd.testing.assert_frame_equal(first.X, second.X)
        assert 130 not in first.positions

    def test_including_frozen_minutes_is_a_deliberate_option(self):
        n = 200
        table = pd.DataFrame({"aggregate_w": series(n).astype("float32"), "washing_machine_w": np.zeros(n, dtype="float32"),
                              "valid": np.r_[np.zeros(100, bool), np.ones(100, bool)], "missing": False})
        assert len(build_house(table, 1).y) == 100
        assert len(build_house(table, 1, exclude_frozen=False).y) == 200


class TestDownsample:
    @staticmethod
    def table(n: int = 120, usable=None) -> pd.DataFrame:
        usable = np.ones(n, bool) if usable is None else usable
        return pd.DataFrame({"official_minute": pd.date_range("2014-05-07", periods=n, freq="min"), "aggregate_w": np.arange(n, dtype="float32"),
                             "washing_machine_w": np.full(n, 60.0, dtype="float32"), "issues_any": False, "valid": usable, "missing": ~usable})

    def test_interval_mean(self):
        out = downsample(self.table(), 30)
        assert len(out) == 4
        assert out["aggregate_w"].iloc[0] == pytest.approx(14.5)
        assert out["washing_machine_w"].iloc[0] == 60

    def test_interval_below_coverage_is_dropped(self):
        usable = np.ones(120, bool)
        usable[:7] = False  # 77% of the first 30 minutes remains.
        out = downsample(self.table(usable=usable), 30)
        assert np.isnan(out["aggregate_w"].iloc[0])
        assert not out["valid"].iloc[0]
        assert out["valid"].iloc[1]

    def test_interval_at_coverage_is_kept_using_usable_minutes_only(self):
        usable = np.ones(120, bool)
        usable[:6] = False  # Exactly 80%.
        out = downsample(self.table(usable=usable), 30)
        assert out["aggregate_w"].iloc[0] == pytest.approx(np.arange(6, 30).mean())
