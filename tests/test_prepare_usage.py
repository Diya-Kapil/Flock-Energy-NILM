"""Official-file preparation and household usage summaries, on small synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import usage
from src.cleaning.prepare import official_to_minutes
from src.cleaning.raw_checks import stamp
from src.config import FROZEN_MIN_MINUTES, SENSORS

BASE = pd.Timestamp("2014-05-07 00:00")  # A Wednesday, well away from clock changes.


def official(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        record = {"Unix": row["unix"], "Issues": row.get("Issues", 0)}
        record.update({s: float(row.get(s, 0.0)) for s in SENSORS})
        records.append(record)
    frame = pd.DataFrame(records)
    frame["Time"] = pd.to_datetime(frame["Unix"], unit="s")
    return frame


DAILY = pd.DataFrame({"day": ["2014-05-07"], "clean_minus_raw_min": [-60], "status": ["verified"]})


class TestOfficialToMinutes:
    def test_mean_uses_the_house_machine_channel(self):
        start = stamp("2014-05-07 12:00")
        rows = [{"unix": start, "Aggregate": 100, "Appliance2": 10, "Appliance5": 999}, {"unix": start + 10, "Aggregate": 200, "Appliance2": 30, "Appliance5": 999}]
        table = official_to_minutes(official(rows), 2, DAILY)
        assert table["aggregate_w"].iloc[0] == 150
        assert table["washing_machine_w"].iloc[0] == 20

    def test_minutes_without_rows_stay_missing(self):
        start = stamp("2014-05-07 12:00")
        table = official_to_minutes(official([{"unix": start, "Aggregate": 100}, {"unix": start + 180, "Aggregate": 100}]), 1, DAILY)
        assert len(table) == 4
        assert table["missing"].tolist() == [False, True, True, False]
        assert table["aggregate_w"].isna().sum() == 2
        assert not table["valid"].iloc[1]

    def test_original_minute_is_kept_and_local_time_is_derived(self):
        table = official_to_minutes(official([{"unix": stamp("2014-05-07 12:00"), "Aggregate": 100}]), 1, DAILY)
        assert table["official_minute"].iloc[0] == pd.Timestamp("2014-05-07 12:00")
        assert table["uk_local_minute"].iloc[0] == pd.Timestamp("2014-05-07 13:00")

    def test_issues_flag_any_row_in_the_minute(self):
        start = stamp("2014-05-07 12:00")
        table = official_to_minutes(official([{"unix": start, "Aggregate": 100, "Issues": 1}, {"unix": start + 10, "Aggregate": 100}]), 1, DAILY)
        assert table["issues_any"].iloc[0]
        assert table["issues_rows"].iloc[0] == 1

    def test_frozen_aggregate_is_flagged_kept_and_not_valid(self):
        start = stamp("2014-05-07 12:00")
        rows = [{"unix": start + i * 60, "Aggregate": 2169} for i in range(FROZEN_MIN_MINUTES)]
        table = official_to_minutes(official(rows), 1, DAILY)
        assert table["aggregate_frozen"].all()
        assert (table["aggregate_w"] == 2169).all()
        assert not table["valid"].any()

    def test_machine_off_is_not_frozen(self):
        start = stamp("2014-05-07 12:00")
        rows = [{"unix": start + i * 60, "Aggregate": 200 + (i % 5), "Appliance5": 0} for i in range(FROZEN_MIN_MINUTES + 5)]
        table = official_to_minutes(official(rows), 1, DAILY)
        assert not table["washing_machine_frozen"].any()
        assert table["valid"].all()

    def test_frozen_machine_above_idle_is_excluded(self):
        start = stamp("2014-05-07 12:00")
        rows = [{"unix": start + i * 60, "Aggregate": 2500 + (i % 5), "Appliance5": 2256} for i in range(FROZEN_MIN_MINUTES)]
        table = official_to_minutes(official(rows), 1, DAILY)
        assert table["washing_machine_frozen"].all()
        assert not table["aggregate_frozen"].any()
        assert not table["valid"].any()

    def test_day_without_an_offset_gets_no_local_time(self):
        table = official_to_minutes(official([{"unix": stamp("2014-06-01 12:00"), "Aggregate": 100}]), 1, DAILY)
        assert pd.isna(table["uk_local_minute"].iloc[0])
        assert table["clock_status"].iloc[0] == "no_offset"


def prepared(power: np.ndarray, aggregate: float = 300.0, start: pd.Timestamp = BASE, offset_minutes: int = 0) -> pd.DataFrame:
    n = len(power)
    minutes = pd.date_range(start, periods=n, freq="min")
    power = np.asarray(power, dtype="float64")
    missing = np.isnan(power)
    return pd.DataFrame({
        "official_minute": minutes, "uk_local_minute": minutes + pd.Timedelta(minutes=offset_minutes),
        "aggregate_w": np.where(missing, np.nan, aggregate + np.nan_to_num(power)).astype("float32"), "washing_machine_w": power.astype("float32"),
        "n_readings": (~missing).astype("int16") * 8, "issues_any": False, "aggregate_over_23kw": np.zeros(n, dtype="int16"), "missing": missing,
        "aggregate_frozen": False, "washing_machine_frozen": False, "frozen": False, "valid": ~missing})


def with_cycle(start_minute: int = 600, minutes: int = 2880, length: int = 60, level: float = 2000.0) -> np.ndarray:
    power = np.zeros(minutes)
    power[start_minute:start_minute + length] = level
    return power


class TestSummaries:
    def test_power_states_shares(self):
        power = np.r_[np.zeros(80), np.full(10, 5.0), np.full(10, 500.0)]
        row = usage.power_states(prepared(power), 1)
        assert (row["zero_pct"], row["standby_pct"], row["active_pct"]) == (80, 10, 10)
        assert row["mean_w_when_active"] == 500

    def test_quality_counts(self):
        power = with_cycle()
        power[:60] = np.nan
        table = prepared(power)
        row = usage.quality_summary(table, 1)
        assert row["missing_minutes"] == 60
        assert row["valid_minutes"] == len(power) - 60

    def test_cycle_start_uses_corrected_local_hour(self):
        table = prepared(with_cycle(start_minute=600), offset_minutes=60)
        cycles = usage.household_cycles(table, 1)
        assert cycles["start_official"].iloc[0] == BASE + pd.Timedelta(minutes=600)
        assert cycles["start_hour"].iloc[0] == 11
        assert cycles["start_weekday"].iloc[0] == 2

    def test_frozen_minutes_do_not_produce_cycles(self):
        table = prepared(with_cycle())
        table.loc[:, "valid"] = False
        table["frozen"] = True
        assert usage.household_cycles(table, 1).empty

    def test_summary_known_answer(self):
        table = prepared(with_cycle(minutes=1440 * 7, length=60))
        cycles = usage.household_cycles(table, 1)
        row = usage.cycle_summary(cycles, table, 1)
        assert row["cycles"] == 1
        assert row["cycles_per_week"] == 1
        assert row["median_energy_kwh"] == pytest.approx(2.0)
        assert row["machine_kwh_per_year"] == pytest.approx(2 / 7 * 365, abs=0.1)
        assert row["energy_in_detected_cycles_pct"] == 100

    def test_hourly_totals_match_cycles(self):
        table = prepared(with_cycle(start_minute=600))
        cycles = usage.household_cycles(table, 1)
        hourly = usage.hourly_usage(cycles, table, 1)
        assert hourly["cycle_starts"].sum() == 1
        assert hourly.loc[hourly["hour"] == 10, "start_share_pct"].iloc[0] == 100
        assert hourly["energy_kwh"].sum() == pytest.approx(2.0)

    def test_hourly_leaves_out_days_without_a_usable_clock(self):
        table = prepared(with_cycle())
        table["uk_local_minute"] = pd.NaT
        cycles = usage.household_cycles(table, 1)
        assert usage.hourly_usage(cycles, table, 1)["cycle_starts"].sum() == 0
        assert usage.clock_sensitivity(cycles, table, 1)["cycles_without_usable_clock"] == 1

    def test_weekday_rate_adjusts_for_days_observed(self):
        table = prepared(with_cycle(minutes=1440 * 7))
        cycles = usage.household_cycles(table, 1)
        weekday = usage.weekday_usage(cycles, table, 1).set_index("weekday")
        assert weekday.loc["Wed", "cycle_starts"] == 1
        assert weekday.loc["Wed", "cycles_per_day"] == 1
        assert weekday.loc["Thu", "cycles_per_day"] == 0

    def test_clock_sensitivity_counts_changed_hours(self):
        table = prepared(with_cycle(start_minute=600), offset_minutes=60)
        row = usage.clock_sensitivity(usage.household_cycles(table, 1), table, 1)
        assert row["cycles_with_hour_changed_pct"] == 100

    def test_frozen_sensitivity_includes_frozen_cycle_only_when_asked(self):
        table = prepared(with_cycle())
        table["frozen"] = table["washing_machine_w"] > 0
        table["valid"] = ~table["frozen"]
        row = usage.frozen_sensitivity(table, 1)
        assert row["cycles_excluding_frozen"] == 0
        assert row["cycles_including_frozen"] == 1

    def test_parameter_sensitivity_lists_settings(self):
        table = prepared(with_cycle())
        out = usage.parameter_sensitivity(table, 1)
        assert "default" in set(out["setting"])
        assert out.loc[out["setting"] == "default", "cycles"].iloc[0] == 1

    def test_representative_day_needs_full_coverage(self):
        table = prepared(with_cycle(minutes=1440 * 3, start_minute=1440 + 600))
        assert usage.representative_day(usage.household_cycles(table, 1), table) == pd.Timestamp("2014-05-08")

    def test_example_cycles_are_deterministic_and_spread(self):
        power = np.zeros(1440 * 3)
        for i, length in enumerate(range(30, 90, 10)):
            power[300 * (i + 1):300 * (i + 1) + length] = 2000
        cycles = usage.household_cycles(prepared(power), 1)
        first, second = usage.example_cycles(cycles, 4), usage.example_cycles(cycles, 4)
        assert first["start_pos"].tolist() == second["start_pos"].tolist()
        assert len(first) == 4
        assert first["duration_min"].is_monotonic_increasing
