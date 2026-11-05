"""Each cleaning rule, tested on small hand-built data with known answers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_part, steady_rows
from src.cleaning import raw_house1 as rh
from src.cleaning.raw_checks import stamp
from src.config import AGGREGATE_REVIEW_W, APPLIANCE_LIMIT_W, FROZEN_MIN_MINUTES, LONG_OUTAGE_MINUTES, SENSORS

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def utc_seconds(minutes: pd.DataFrame) -> np.ndarray:
    return ((minutes["timestamp_utc"] - EPOCH) // pd.Timedelta(seconds=1)).to_numpy()


def one_part_table(rows: list[dict]) -> pd.DataFrame:
    """Clean a single block of rows, using a one-row Part 2 far later to satisfy the API."""
    later = make_part([{"clock": "2015-01-10 12:00", "Aggregate": 200}], 2)
    return rh.clean_house1(make_part(rows, 1), later)


# ----------------------------------------------------------------- clock
class TestAssignUtc:
    def test_winter_rows_equal_utc(self):
        out = rh.assign_utc(make_part([{"clock": "2014-01-10 12:00"}], 1))
        assert out["utc"].iloc[0] == stamp("2014-01-10 12:00")

    def test_summer_rows_are_one_hour_ahead_of_utc(self):
        out = rh.assign_utc(make_part([{"clock": "2014-07-10 12:00"}], 1))
        assert out["utc"].iloc[0] == stamp("2014-07-10 11:00")

    def test_original_timestamps_are_preserved(self):
        frame = make_part([{"clock": "2014-07-10 12:00"}], 1)
        out = rh.assign_utc(frame)
        assert out["Unix"].iloc[0] == frame["Unix"].iloc[0]
        assert out["Time"].iloc[0] == frame["Time"].iloc[0]

    def test_spring_change_leaves_no_gap_in_utc(self):
        out = rh.assign_utc(make_part([{"clock": "2014-03-30 00:59:50"}, {"clock": "2014-03-30 02:00:00"}], 1))
        assert out["utc"].iloc[1] - out["utc"].iloc[0] == 10

    def test_reading_inside_the_missing_spring_hour_is_an_error(self):
        with pytest.raises(Exception, match="(?i)nonexistent|2014-03-30"):
            rh.assign_utc(make_part([{"clock": "2014-03-30 01:30:00"}], 1))

    def test_autumn_hour_separates_on_backward_step(self):
        rows = [{"clock": "2013-10-27 01:59:50"}, {"clock": "2013-10-27 01:00:10"}, {"clock": "2013-10-27 01:00:20"}]
        out = rh.assign_utc(make_part(rows, 1))
        assert out["utc"].tolist() == [stamp("2013-10-27 00:59:50"), stamp("2013-10-27 01:00:10"), stamp("2013-10-27 01:00:20")]
        assert not out["dst_ambiguous"].any()

    def test_autumn_hour_without_a_step_is_marked_ambiguous(self):
        rows = [{"clock": "2014-10-26 01:10:00"}, {"clock": "2014-10-26 01:10:03"}]
        out = rh.assign_utc(make_part(rows, 2))
        assert out["dst_ambiguous"].all()
        assert out["utc"].iloc[0] == stamp("2014-10-26 00:10:00")


class TestClockInTable:
    def test_grid_is_continuous_across_spring_change(self):
        rows = steady_rows("2014-03-30 00:30", 30) + steady_rows("2014-03-30 02:00", 30)
        minutes = one_part_table(rows).minutes
        span = minutes[minutes["timestamp_utc"] < "2014-03-30 01:30:00+00:00"]
        assert len(span) == 60
        assert (span["timestamp_utc"].diff().dropna() == pd.Timedelta(minutes=1)).all()
        assert not span["Aggregate_missing"].any()

    def test_spring_change_is_not_counted_as_an_outage(self):
        rows = steady_rows("2014-03-30 00:30", 30) + steady_rows("2014-03-30 02:00", 30)
        minutes = one_part_table(rows).minutes
        assert (minutes[minutes["timestamp_utc"] < "2014-03-30 01:30:00+00:00"]["outage_minutes"] == 0).all()

    def test_raw_clock_label_is_kept(self):
        minutes = one_part_table(steady_rows("2014-07-10 12:00", 5)).minutes
        assert minutes["raw_clock_minute"].iloc[0] == pd.Timestamp("2014-07-10 12:00")
        assert minutes["utc_offset_min"].iloc[0] == 60

    def test_unseparated_autumn_second_pass_is_missing_flagged_and_not_an_outage(self):
        part1 = make_part(steady_rows("2014-10-25 23:00", 10), 1)
        interleaved = steady_rows("2014-10-26 00:30", 60) + steady_rows("2014-10-26 01:00", 60) + steady_rows("2014-10-26 02:00", 10)
        part2 = make_part(sorted(interleaved, key=lambda row: row["unix"]), 2)  # Part 2 is sorted, so passes interleave.
        minutes = rh.clean_house1(part1, part2).minutes
        second = minutes[(minutes["timestamp_utc"] >= "2014-10-26 01:00:00+00:00")
                         & (minutes["timestamp_utc"] < "2014-10-26 02:00:00+00:00")]
        assert len(second) == 60
        assert second["Aggregate_missing"].all()
        assert second["dst_ambiguous"].all()
        assert (second["outage_minutes"] == 0).all()


# ----------------------------------------------------------------- source and duplicates
class TestSourceSelection:
    def test_part1_is_used_strictly_before_first_part2_timestamp(self):
        part1 = rh.assign_utc(make_part(steady_rows("2014-01-10 12:00", 30), 1))
        part2 = rh.assign_utc(make_part(steady_rows("2014-01-10 12:20", 30), 2))
        combined, switch = rh.select_source(part1, part2)
        assert switch == stamp("2014-01-10 12:20")
        assert combined.loc[combined["part"] == 1, "utc"].max() < switch
        assert combined.loc[combined["part"] == 2, "utc"].min() == switch

    def test_part1_rows_after_switch_are_counted_as_unused(self, small_parts):
        flow = rh.clean_house1(*small_parts).flow
        assert flow["part1_rows_kept_before_switch"] + flow["part1_rows_not_used_after_switch"] == flow["part1_rows_read"]

    def test_values_after_switch_come_from_part2(self):
        part1 = make_part(steady_rows("2014-01-10 12:00", 30, Appliance5=100), 1)
        part2 = make_part(steady_rows("2014-01-10 12:20", 30, Appliance5=300), 2)
        minutes = rh.clean_house1(part1, part2).minutes
        after = minutes[minutes["raw_clock_minute"] >= "2014-01-10 12:20"]
        assert (after["Appliance5"] == 300).all()
        assert (after["source_part"] == 2).all()


class TestExactDuplicates:
    def test_identical_rows_collapse_to_one(self):
        rows = [{"clock": "2014-01-10 12:00:00", "Aggregate": 200}] * 3 + [{"clock": "2014-01-10 12:00:10", "Aggregate": 210}]
        out = rh.drop_exact_duplicates(rh.assign_utc(make_part(rows, 1)))
        assert len(out) == 2

    def test_rows_differing_only_in_one_sensor_are_kept(self):
        rows = [{"clock": "2014-01-10 12:00:00", "Aggregate": 200}, {"clock": "2014-01-10 12:00:00", "Aggregate": 205}]
        assert len(rh.drop_exact_duplicates(rh.assign_utc(make_part(rows, 1)))) == 2

    def test_exact_duplicates_do_not_change_the_minute_mean(self):
        rows = [{"clock": "2014-01-10 12:00:00", "Aggregate": 100}] * 5 + [{"clock": "2014-01-10 12:00:10", "Aggregate": 200}]
        minutes = one_part_table(rows).minutes
        assert minutes["Aggregate"].iloc[0] == pytest.approx(150)


class TestConflicts:
    def test_conflicting_readings_are_averaged_so_neither_is_lost(self):
        rows = [{"clock": "2014-01-10 12:00:00", "Appliance5": 3}, {"clock": "2014-01-10 12:00:00", "Appliance5": 2372}]
        merged = rh.merge_timestamps(rh.assign_utc(make_part(rows, 1)))
        assert merged["Appliance5"].iloc[0] == pytest.approx((3 + 2372) / 2)

    def test_a_nan_never_erases_another_rows_reading(self):
        rows = [{"clock": "2014-01-10 12:00:00", "Aggregate": 200, "Appliance5": None},
                {"clock": "2014-01-10 12:00:00", "Aggregate": None, "Appliance5": 50}]
        merged = rh.merge_timestamps(rh.assign_utc(make_part(rows, 2)))
        assert merged["Aggregate"].iloc[0] == 200
        assert merged["Appliance5"].iloc[0] == 50

    def test_every_conflict_is_logged_with_its_resolution(self):
        rows = [{"clock": "2014-01-10 12:00:00", "Appliance5": 10}, {"clock": "2014-01-10 12:00:00", "Appliance5": 30}]
        result = one_part_table(rows)
        log = result.conflicts[result.conflicts["sensor"] == "Appliance5"]
        assert len(log) == 1
        assert log["first_value"].iloc[0] == 10
        assert log["last_value"].iloc[0] == 30
        assert log["resolved_w"].iloc[0] == 20
        assert result.minutes["n_conflicts"].iloc[0] == 1

    def test_agreeing_rows_are_not_logged_as_conflicts(self):
        rows = [{"clock": "2014-01-10 12:00:00", "Aggregate": 200, "Appliance5": None},
                {"clock": "2014-01-10 12:00:00", "Aggregate": 200, "Appliance5": 40}]
        assert len(one_part_table(rows).conflicts) == 0

    def test_spike_does_not_influence_a_resolved_value(self):
        rows = [{"clock": "2014-01-10 12:00:00", "Appliance5": 100}, {"clock": "2014-01-10 12:00:00", "Appliance5": 8192}]
        result = one_part_table(rows)
        log = result.conflicts[result.conflicts["sensor"] == "Appliance5"]
        assert log["involves_spike"].iloc[0]
        assert log["resolved_w"].iloc[0] == 100


# ----------------------------------------------------------------- outliers
class TestSpikes:
    def test_reading_above_limit_becomes_missing_not_zero(self):
        rows = [{"clock": "2014-01-10 12:00:00", "Appliance5": APPLIANCE_LIMIT_W + 1}]
        minutes = one_part_table(rows).minutes
        assert np.isnan(minutes["Appliance5"].iloc[0])
        assert minutes["Appliance5_missing"].iloc[0]
        assert minutes["Appliance5_spikes"].iloc[0] == 1

    def test_reading_exactly_at_limit_is_kept(self):
        rows = [{"clock": "2014-01-10 12:00:00", "Appliance5": APPLIANCE_LIMIT_W}]
        minutes = one_part_table(rows).minutes
        assert minutes["Appliance5"].iloc[0] == APPLIANCE_LIMIT_W
        assert minutes["Appliance5_spikes"].iloc[0] == 0

    def test_other_readings_in_the_minute_survive(self):
        rows = [{"clock": "2014-01-10 12:00:00", "Appliance5": 200},
                {"clock": "2014-01-10 12:00:10", "Appliance5": 16384},
                {"clock": "2014-01-10 12:00:20", "Appliance5": 400}]
        minutes = one_part_table(rows).minutes
        assert minutes["Appliance5"].iloc[0] == pytest.approx(300)
        assert minutes["Appliance5_n"].iloc[0] == 2

    def test_aggregate_is_not_limited_by_the_appliance_rule(self):
        rows = [{"clock": "2014-01-10 12:00:00", "Aggregate": 9000}]
        assert one_part_table(rows).minutes["Aggregate"].iloc[0] == 9000


class TestAggregateReview:
    def test_reading_above_review_level_is_kept_and_flagged(self):
        rows = [{"clock": "2014-01-10 12:00:00", "Aggregate": AGGREGATE_REVIEW_W + 1000}]
        minutes = one_part_table(rows).minutes
        assert minutes["Aggregate"].iloc[0] == AGGREGATE_REVIEW_W + 1000
        assert minutes["Aggregate_over_23kw"].iloc[0] == 1

    def test_reading_at_review_level_is_not_flagged(self):
        rows = [{"clock": "2014-01-10 12:00:00", "Aggregate": AGGREGATE_REVIEW_W}]
        assert one_part_table(rows).minutes["Aggregate_over_23kw"].iloc[0] == 0


# ----------------------------------------------------------------- minutes and gaps
class TestMinutes:
    def test_mean_uses_available_readings_only(self):
        rows = [{"clock": "2014-01-10 12:00:00", "Appliance5": 100},
                {"clock": "2014-01-10 12:00:10", "Appliance5": None},
                {"clock": "2014-01-10 12:00:20", "Appliance5": 200}]
        minutes = one_part_table(rows).minutes
        assert minutes["Appliance5"].iloc[0] == pytest.approx(150)
        assert minutes["Appliance5_n"].iloc[0] == 2
        assert minutes["Aggregate_n"].iloc[0] == 3

    def test_nothing_is_interpolated_or_filled_across_a_gap(self):
        rows = steady_rows("2014-01-10 12:00", 2, Appliance5=500) + steady_rows("2014-01-10 12:10", 2, Appliance5=500)
        minutes = one_part_table(rows).minutes
        gap = minutes[(minutes["raw_clock_minute"] >= "2014-01-10 12:02") & (minutes["raw_clock_minute"] < "2014-01-10 12:10")]
        assert len(gap) == 8
        for sensor in SENSORS:
            assert gap[sensor].isna().all()
            assert gap[f"{sensor}_missing"].all()
            assert (gap[f"{sensor}_n"] == 0).all()

    def test_silent_sensor_is_not_held_at_its_last_value(self):
        rows = steady_rows("2014-01-10 12:00", 2, Appliance5=2000) + [
            {"unix": stamp("2014-01-10 12:02") + i * 10, "Aggregate": 200, "Appliance5": None} for i in range(30)]
        minutes = one_part_table(rows).minutes
        assert minutes["Appliance5"].iloc[2:7].isna().all()
        assert minutes["Aggregate"].iloc[2:7].notna().all()

    def test_part1_zero_is_preserved_not_treated_as_missing(self):
        minutes = one_part_table(steady_rows("2014-01-10 12:00", 2, Appliance5=0)).minutes
        assert (minutes["Appliance5"].iloc[:2] == 0).all()
        assert not minutes["Appliance5_missing"].iloc[:2].any()
        assert minutes["source_part"].iloc[0] == 1

    def test_missing_flag_always_matches_a_missing_value(self, small_parts):
        minutes = rh.clean_house1(*small_parts).minutes
        for sensor in SENSORS:
            assert (minutes[sensor].isna() == minutes[f"{sensor}_missing"]).all()


class TestOutages:
    def test_short_gap_is_recorded_but_not_a_long_outage(self):
        rows = steady_rows("2014-01-10 12:00", 2) + steady_rows("2014-01-10 12:04", 2)
        minutes = one_part_table(rows).minutes
        gap = minutes[(minutes["raw_clock_minute"] >= "2014-01-10 12:02") & (minutes["raw_clock_minute"] < "2014-01-10 12:04")]
        assert (gap["outage_minutes"] == 2).all()
        assert not gap["long_outage"].any()

    def test_gap_at_threshold_is_flagged_long_and_stays_missing(self):
        n = LONG_OUTAGE_MINUTES
        rows = steady_rows("2014-01-10 12:00", 2) + steady_rows(str(pd.Timestamp("2014-01-10 12:02") + pd.Timedelta(minutes=n)), 2)
        minutes = one_part_table(rows).minutes
        gap = minutes.iloc[2:2 + n]
        assert gap["long_outage"].all()
        assert (gap["outage_minutes"] == n).all()
        assert gap["Aggregate"].isna().all()

    def test_sensor_silence_is_not_a_whole_house_outage(self):
        rows = [{"unix": stamp("2014-01-10 12:00") + i * 10, "Aggregate": 200 + i, "Appliance5": None} for i in range(60)]
        rows.append({"unix": stamp("2014-01-10 12:10"), "Aggregate": 200, "Appliance5": 10})
        minutes = one_part_table(rows).minutes.iloc[:10]
        assert (minutes["outage_minutes"] == 0).all()
        assert minutes["Appliance5_missing"].all()
        assert (minutes["Appliance5_gap_minutes"] == 10).all()

    def test_run_lengths_known_answer(self):
        result = rh.run_lengths(np.array([False, True, True, False, True]))
        assert result.tolist() == [0, 2, 2, 0, 1]


class TestFrozen:
    def test_long_identical_run_is_flagged_and_value_kept(self):
        n = FROZEN_MIN_MINUTES
        grid = np.arange(n + 5)
        values = np.r_[np.arange(5.0), np.full(n, 2256.0)]
        mask = rh.frozen_mask(grid, values, None)
        assert mask[5:].all()
        assert not mask[:5].any()

    def test_run_shorter_than_threshold_is_not_flagged(self):
        n = FROZEN_MIN_MINUTES - 1
        assert not rh.frozen_mask(np.arange(n), np.full(n, 100.0), None).any()

    def test_appliance_off_is_not_frozen(self):
        n = FROZEN_MIN_MINUTES + 10
        assert not rh.frozen_mask(np.arange(n), np.zeros(n), 20).any()

    def test_missing_minutes_break_a_run(self):
        n = FROZEN_MIN_MINUTES // 2 + 5
        values = np.r_[np.full(n, 100.0), np.nan, np.full(n, 100.0)]
        assert not rh.frozen_mask(np.arange(len(values)), values, None).any()

    def test_table_keeps_frozen_values(self):
        rows = [{"unix": stamp("2014-01-10 12:00") + i * 10, "Aggregate": 12505, "Appliance5": 2256}
                for i in range((FROZEN_MIN_MINUTES + 1) * 6)]
        minutes = one_part_table(rows).minutes
        first = minutes.iloc[: FROZEN_MIN_MINUTES]
        assert first["Aggregate_frozen"].all()
        assert first["Appliance5_frozen"].all()
        assert (first["Aggregate"] == 12505).all()


class TestWholePipeline:
    def test_result_is_deterministic(self, small_parts):
        first = rh.clean_house1(*small_parts).minutes
        second = rh.clean_house1(*small_parts).minutes
        pd.testing.assert_frame_equal(first, second)

    def test_inputs_are_not_modified(self, small_parts):
        before = [p.copy() for p in small_parts]
        rh.clean_house1(*small_parts)
        for original, now in zip(before, small_parts):
            pd.testing.assert_frame_equal(original, now)

    def test_row_counts_add_up(self, small_parts):
        flow = rh.clean_house1(*small_parts).flow
        assert flow["rows_after_exact_duplicates"] == flow["rows_after_source_selection"] - flow["exact_duplicate_rows_removed"]
        assert flow["unique_timestamps"] == flow["rows_after_exact_duplicates"] - flow["rows_merged_into_an_existing_timestamp"]
