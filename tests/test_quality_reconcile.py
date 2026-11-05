"""Report statistics and the comparison with the official file, on known answers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import make_part, steady_rows
from src.cleaning import quality, raw_house1 as rh, reconcile
from src.cleaning.raw_checks import stamp
from src.cleaning.report import table
from src.config import SENSORS


def result_with_gap():
    rows = steady_rows("2014-01-10 12:00", 5, Appliance5=100) + steady_rows("2014-01-10 12:15", 5, Appliance5=100)
    part1 = make_part(rows, 1)
    part2 = make_part(steady_rows("2014-01-10 12:30", 5, Appliance5=100), 2)
    return part1, part2, rh.clean_house1(part1, part2)


class TestOutageProfile:
    def test_known_gaps(self):
        _, _, result = result_with_gap()
        out = quality.outage_profile(result.minutes)
        assert out["outages_of_any_length"] == 2
        assert out["long_outages"] == 2
        assert out["whole_house_outage_minutes"] == 10 + 10
        assert sum(b["outages"] for b in out["buckets"]) == 2

    def test_short_outages_appear_in_a_bucket(self):
        rows = steady_rows("2014-01-10 12:00", 2) + steady_rows("2014-01-10 12:03", 2)
        minutes = rh.clean_house1(make_part(rows, 1), make_part(steady_rows("2014-01-10 13:00", 1), 2)).minutes
        out = quality.outage_profile(minutes)
        assert next(b for b in out["buckets"] if b["length"] == "1 to 2 min")["outages"] >= 1


class TestMissingProfile:
    def test_counts_add_up_to_the_grid(self):
        _, _, result = result_with_gap()
        for row in quality.missing_profile(result.minutes)[:-1]:
            assert row["minutes_present"] + row["minutes_missing"] == len(result.minutes)


class TestSpikeProfile:
    def test_bit_patterns_are_recognised(self):
        rows = [{"clock": "2014-01-10 12:00:00", "Appliance5": 8192}, {"clock": "2014-01-10 12:00:10", "Appliance5": 5000}]
        result = rh.clean_house1(make_part(rows, 1), make_part(steady_rows("2014-01-10 13:00", 1), 2))
        row = next(r for r in quality.spike_profile(result.rows, result.minutes) if r["sensor"] == "Appliance5")
        assert row["readings_over_limit"] == 2
        assert row["bit_pattern_pct"] == 50
        assert row["minutes_left_with_no_reading"] == 1


class TestAggregateReview:
    def test_isolated_reading_is_told_apart_from_sustained_load(self):
        isolated = steady_rows("2014-01-10 12:00", 2)
        isolated[6]["Aggregate"] = 29000
        sustained = [{"unix": stamp("2014-01-10 13:00") + i * 10, "Aggregate": 24000 if i in (3, 4) else 20000} for i in range(10)]
        result = rh.clean_house1(make_part(isolated + sustained, 1), make_part(steady_rows("2014-01-10 14:00", 1), 2))
        review = quality.aggregate_review(result.rows, result.minutes)
        assert review["readings_over_threshold"] == 3
        assert review["isolated_readings"] == 1
        assert review["readings_at_exactly_24000_w"] == 2
        assert sorted(e["kind"] for e in review["episodes"]) == ["isolated single reading", "sustained high load"]


class TestDstProfile:
    def test_reports_each_transition(self):
        _, _, result = result_with_gap()
        assert len(quality.dst_profile(result.minutes)) == 4


class TestFrozenProfile:
    def test_energy_and_share_known_answer(self):
        rows = [{"unix": stamp("2014-01-10 12:00") + i * 10, "Aggregate": 300, "Appliance5": 1000} for i in range(61 * 6)]
        rows += [{"unix": stamp("2014-01-10 13:01") + i * 10, "Aggregate": 200 + (i % 7), "Appliance5": 0} for i in range(60)]
        result = rh.clean_house1(make_part(rows, 1), make_part(steady_rows("2014-01-10 15:00", 1), 2))
        row = next(r for r in quality.frozen_profile(result.minutes)["per_sensor"] if r["sensor"] == "Appliance5")
        assert row["runs"] == 1
        assert row["minutes"] == 61
        assert row["share_of_sensor_energy_pct"] == 100


class TestPartProfile:
    def test_reports_gaps_and_missing(self):
        rows = [{"clock": "2014-01-10 12:00:00", "Appliance5": None}, {"clock": "2014-01-10 12:05:00", "Appliance5": 10}]
        out = quality.part_profile(make_part(rows, 2))
        assert out["gaps_over_2min"] == 1
        assert next(s for s in out["sensors"] if s["sensor"] == "Appliance5")["missing_rows"] == 1


class TestReconcile:
    @staticmethod
    def official_from(minutes: pd.DataFrame) -> pd.DataFrame:
        """An official-style frame with the same values, one row per minute."""
        keep = minutes[~minutes["Aggregate_missing"]]
        frame = pd.DataFrame({sensor: keep[sensor].astype("float64").to_numpy() for sensor in SENSORS})
        frame.insert(0, "Unix", ((keep["raw_clock_minute"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy())
        return frame

    def test_identical_data_agrees_perfectly(self):
        _, _, result = result_with_gap()
        minutes = result.minutes
        official = self.official_from(minutes)
        daily = pd.DataFrame({"day": ["2014-01-10"], "clean_minus_raw_min": [0], "status": ["verified"]})
        ours = reconcile.ours_on_raw_clock(minutes)
        theirs = reconcile.official_on_raw_clock(official, daily)
        lo = stamp("2014-01-10 12:00")
        row = next(r for r in reconcile.compare_period(ours, theirs, lo, lo + 3600) if r["sensor"] == "Aggregate")
        assert row["mean_abs_diff_w"] == 0
        assert row["minutes_only_ours"] == 0
        assert row["minutes_only_official"] == 0

    def test_official_offset_is_corrected_before_comparing(self):
        _, _, result = result_with_gap()
        official = self.official_from(result.minutes)
        official["Unix"] = official["Unix"] - 3600
        daily = pd.DataFrame({"day": ["2014-01-10"], "clean_minus_raw_min": [-60], "status": ["verified"]})
        theirs = reconcile.official_on_raw_clock(official, daily)
        assert theirs.index.min() == stamp("2014-01-10 12:00") // 60

    def test_spike_treatment_counts_official_zero(self):
        rows = [{"clock": "2014-01-10 12:00:00", "Appliance5": 8192}]
        minutes = rh.clean_house1(make_part(rows, 1), make_part(steady_rows("2014-01-10 13:00", 1), 2)).minutes
        ours = reconcile.ours_on_raw_clock(minutes)
        official = pd.DataFrame(0.0, index=ours.index, columns=SENSORS)
        row = next(r for r in reconcile.spike_treatment(ours, official) if r["sensor"] == "Appliance5")
        assert row["minutes_with_spike_removed"] == 1
        assert row["official_exactly_zero_minutes"] == 1

    def test_held_values_are_found(self):
        rows = [{"unix": stamp("2014-01-10 12:00") + i * 10, "Aggregate": 200, "Appliance5": 50 if i < 6 else None} for i in range(30)]
        minutes = rh.clean_house1(make_part([{"clock": "2014-01-10 11:00", "Aggregate": 200}], 1), make_part(rows, 2)).minutes
        ours = reconcile.ours_on_raw_clock(minutes)
        official = pd.DataFrame(50.0, index=ours.index, columns=SENSORS)
        row = next(r for r in reconcile.held_values(ours, official, stamp("2014-01-10 12:00") // 60) if r["sensor"] == "Appliance5")
        assert row["silent_minutes_here"] == 4
        assert row["official_value_is_nonzero"] == 4


class TestSilenceBehaviour:
    @staticmethod
    def build():
        start = stamp("2014-01-10 12:00")
        rows = [{"unix": start + i * 10, "Aggregate": 200 + (i % 5), "Appliance5": 50 if i < 6 else None} for i in range(70 * 6)]
        rows.append({"unix": start + 70 * 60, "Aggregate": 200, "Appliance5": 50})
        minutes = rh.clean_house1(make_part([{"clock": "2014-01-10 11:00", "Aggregate": 200}], 1), make_part(rows, 2)).minutes
        return reconcile.ours_on_raw_clock(minutes), start // 60

    def test_held_nonzero_value_is_counted(self):
        ours, start = self.build()
        official = pd.DataFrame(50.0, index=ours.index, columns=SENSORS)
        out = reconcile.silence_behaviour(ours, official, start)
        longest = next(r for r in out if r["silence_length"] == "60 min or more")
        assert longest == {"silence_length": "60 min or more", "runs": 1, "held_one_value": 1, "held_at_nonzero": 1}

    def test_changing_official_values_are_not_counted_as_held(self):
        ours, start = self.build()
        official = pd.DataFrame(np.arange(len(ours), dtype="float64").reshape(-1, 1).repeat(len(SENSORS), axis=1), index=ours.index, columns=SENSORS)
        longest = next(r for r in reconcile.silence_behaviour(ours, official, start) if r["silence_length"] == "60 min or more")
        assert longest["runs"] == 1
        assert longest["held_one_value"] == 0


class TestReportTable:
    def test_formats_values(self):
        out = table([{"a": 1234, "b": 1.5, "c": None, "d": True}], [("a", "A"), ("b", "B"), ("c", "C"), ("d", "D")])
        assert "| 1,234 | 1.5 | n/a | yes |" in out

    def test_nan_prints_as_not_available(self):
        assert "n/a" in table([{"a": float("nan")}], [("a", "A")])
