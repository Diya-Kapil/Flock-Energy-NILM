"""The duplicate-conflict and part-boundary checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_part, steady_rows
from src.cleaning import raw_checks as rc
from src.cleaning.raw_checks import stamp


def rows_at(clock: str, *values: float) -> list[dict]:
    return [{"clock": clock, "Appliance5": value} for value in values]


class TestFindConflicts:
    def test_different_readings_at_one_timestamp_conflict(self):
        found = rc.find_conflicts(make_part(rows_at("2014-01-10 12:00:00", 3, 2372), 1))
        row = found[found["sensor"] == "Appliance5"].iloc[0]
        assert (row["first_value"], row["last_value"], row["spread_w"]) == (3, 2372, 2369)
        assert row["kind"] == "both_nonzero"

    def test_zero_against_a_reading_is_classified(self):
        found = rc.find_conflicts(make_part(rows_at("2014-01-10 12:00:00", 0, 150), 1))
        assert found[found["sensor"] == "Appliance5"]["kind"].iloc[0] == "zero_vs_nonzero"

    def test_nan_is_not_a_conflict(self):
        rows = [{"clock": "2014-01-10 12:00:00", "Appliance5": None}, {"clock": "2014-01-10 12:00:00", "Appliance5": 50}]
        assert rc.find_conflicts(make_part(rows, 2)).empty

    def test_different_timestamps_never_conflict(self):
        rows = rows_at("2014-01-10 12:00:00", 3) + rows_at("2014-01-10 12:00:10", 2372)
        assert rc.find_conflicts(make_part(rows, 1)).empty

    def test_spike_is_marked_and_limit_removes_the_conflict(self):
        frame = make_part(rows_at("2014-01-10 12:00:00", 100, 8192), 1)
        assert rc.find_conflicts(frame)["involves_spike"].any()
        assert rc.find_conflicts(frame, limit=4000).empty


class TestNeighbourEvidence:
    def test_real_change_matches_neighbours_either_side(self):
        rows = (rows_at("2014-01-10 12:00:00", 10) + rows_at("2014-01-10 12:00:10", 10, 90) + rows_at("2014-01-10 12:00:20", 90))
        result = rc.neighbour_evidence(make_part(rows, 1), "Appliance5")
        assert result["conflicts"] == 1
        assert result["first_equals_previous_pct"] == 100
        assert result["last_equals_next_pct"] == 100

    def test_no_conflicts_reports_zero(self):
        assert rc.neighbour_evidence(make_part(rows_at("2014-01-10 12:00:00", 5), 1), "Appliance5")["conflicts"] == 0


class TestRuleImpact:
    def test_mean_rule_lies_between_first_and_last(self):
        rows = rows_at("2014-01-10 12:00:00", 0, 100)
        result = rc.rule_impact(make_part(rows, 1), "Appliance5")
        assert result["minutes_with_a_conflict"] == 1
        assert result["last_vs_mean_abs_w_max"] == 50
        assert result["first_vs_last_abs_w_max"] == 100


class TestConflictTimeSplit:
    def test_rows_one_second_apart_under_one_label_are_counted(self):
        frame = make_part(rows_at("2014-01-10 12:00:00", 3, 2372), 1)
        frame.loc[1, "Time"] = frame.loc[0, "Time"] + pd.Timedelta(seconds=1)
        out = rc.conflict_time_split(frame, rc.find_conflicts(frame))
        assert out == {"timestamps_with_a_conflict": 1, "with_differing_time_pct": 100.0}


class TestIndependentReadings:
    def test_bursts_count_as_one_poll(self):
        start = stamp("2014-01-10 12:00")
        offsets = [0, 1, 2, 15, 16, 30, 31, 45]
        out = rc.independent_readings(make_part([{"unix": start + o} for o in offsets], 1))
        assert out["rows_per_minute_median"] == 8
        assert out["polls_per_minute_median"] == 4


class TestNanBehaviour:
    def test_known_silence_and_rates(self):
        start = stamp("2014-01-10 12:00")
        values = [0, None, 0, 100, None, 100, 0, 0]
        rows = [{"unix": start + i * 10, "Appliance5": v} for i, v in enumerate(values)]
        out = rc.nan_behaviour(make_part(rows, 2), "Appliance5")
        short = out["buckets"][0]
        assert short["silences"] == 2
        assert short["same_value_both_sides_pct"] == 100
        assert out["silence_seconds_p50"] == 20
        assert out["nan_rate_after_active_reading_pct"] == 50
        assert out["nan_rate_after_off_reading_pct"] == pytest.approx(33.33)  # 3 off readings have a next row; 1 is NaN.


class TestOverview:
    def test_counts(self):
        rows = [{"clock": "2014-01-10 12:00:00", "Aggregate": 200}, {"clock": "2014-01-10 12:00:00", "Aggregate": 200},
                {"clock": "2014-01-10 12:00:00", "Aggregate": 210}, {"clock": "2014-01-10 12:00:10", "Aggregate": 220}]
        out = rc.duplicate_overview(make_part(rows, 1))
        assert out["rows"] == 4
        assert out["unique_timestamps"] == 2
        assert out["extra_rows_beyond_first"] == 2
        assert out["rows_identical_to_an_earlier_row"] == 1
        assert out["max_rows_on_one_timestamp"] == 3


def varied_minutes(start: str, minutes: int, seed: int = 1) -> pd.DataFrame:
    """A random-looking minute table, so a lag between two copies is detectable."""
    rng = np.random.default_rng(seed)
    index = np.arange(stamp(start) // 60, stamp(start) // 60 + minutes)
    values = 200 + rng.normal(0, 50, minutes).cumsum() % 400
    return pd.DataFrame({sensor: values + i * 3 for i, sensor in enumerate(rc.SENSORS)}, index=index)


class TestBoundary:
    def test_daily_lag_finds_a_known_offset(self):
        a = varied_minutes("2014-07-01", 3 * 1440)
        b = a.copy()
        b.index = b.index + 5
        table = rc.daily_part_lag(a, b, range(-10, 11))
        assert table[table["minutes_compared"] > 1000]["part2_lag_min"].eq(5).all()

    def test_channel_agreement_detects_a_swapped_channel(self):
        rng = np.random.default_rng(3)
        index = np.arange(stamp("2014-07-01") // 60, stamp("2014-07-01") // 60 + 1440)
        a = pd.DataFrame({sensor: rng.normal(100, 20, 1440) for sensor in rc.SENSORS}, index=index)
        b = a.rename(columns={"Appliance1": "Appliance2", "Appliance2": "Appliance1"})
        table = rc.channel_agreement(a, b, stamp("2014-07-01"), stamp("2014-07-02")).set_index("sensor")
        assert table.loc["Appliance1", "best_part1_channel_for_part2"] == "Appliance2"
        assert table.loc["Appliance5", "best_part1_channel_for_part2"] == "Appliance5"

    def test_availability_counts_and_lists_part1_only_runs(self):
        grid = np.arange(stamp("2014-07-01") // 60, stamp("2014-07-01") // 60 + 100)
        m1 = pd.DataFrame({"Aggregate": 1.0}, index=grid[:80])
        m2 = pd.DataFrame({"Aggregate": 1.0}, index=grid[50:])
        out = rc.part_availability(m1, m2, stamp("2014-07-01"), stamp("2014-07-01") + 6000)
        assert (out["both"], out["part1_only"], out["part2_only"], out["neither"]) == (30, 50, 20, 0)
        assert out["longest_part1_only_runs"] == [("2014-07-01 00:00:00", 50)]

    def test_switch_continuity_reports_the_step(self):
        p1 = make_part(steady_rows("2014-07-01 10:00", 120), 1)
        p2 = make_part(steady_rows("2014-07-01 12:00", 60), 2)
        out = rc.switch_continuity(p1, p2, rc.minute_table(p1), rc.minute_table(p2))
        assert out["first_part2_timestamp"] == "2014-07-01 12:00:00"
        assert out["seconds_between"] == 10
        assert out["step_at_switch_w"] < 5


class TestSnapRowRates:
    def test_counts_rows_in_each_20_minute_block(self):
        start = stamp("2014-08-07 11:00")
        p1 = make_part([{"unix": start + i * 60} for i in range(40)], 1)
        p2 = make_part([{"unix": start + i * 120} for i in range(20)], 2)
        out = rc.snap_row_rates(p1, p2)
        assert out["part1"] == [20, 20]
        assert out["part2"] == [10, 10]


class TestWithinPartAlignment:
    def test_lag_zero_when_sensors_share_a_clock(self):
        rng = np.random.default_rng(5)
        index = np.arange(stamp("2014-06-28") // 60, stamp("2014-10-01") // 60)
        machine = np.abs(rng.normal(0, 100, len(index)))
        table = pd.DataFrame({"Appliance5": machine, "Aggregate": machine + 200}, index=index)
        rows = rc.within_part_alignment(table, table)
        assert all(r["part1_lag_min"] == 0 and r["part2_lag_min"] == 0 for r in rows)


class TestClockChanges:
    def test_backward_step_is_found(self):
        rows = [{"clock": "2013-10-27 01:59:58"}, {"clock": "2013-10-27 01:00:01"}]
        steps = rc.backward_steps(make_part(rows, 1))
        assert steps == [{"file_row": 0, "from": "2013-10-27 01:59:58", "to": "2013-10-27 01:00:01", "seconds": -3597}]

    def test_rows_by_clock_hour(self):
        rows = [{"clock": "2014-03-30 00:30:00"}, {"clock": "2014-03-30 02:30:00"}, {"clock": "2014-03-30 03:30:00"}]
        result = rc.transition_rows(make_part(rows, 1), {"2014-03-30": "forward"})
        assert result[0]["rows_in_clock_hours_00_to_03"] == [1, 0, 1, 1]
