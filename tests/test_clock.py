"""The official-clock measurement and the derived UK local time."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cleaning import clock
from src.cleaning.raw_checks import stamp


def noise_series(start: str, days: int, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    first = stamp(start) // 60
    return pd.Series(np.abs(rng.normal(300, 120, days * 1440)), index=np.arange(first, first + days * 1440))


def shifted(series: pd.Series, minutes: int) -> pd.Series:
    out = series.copy()
    out.index = out.index + minutes
    return out


def daily_table(rows: list[tuple[str, float, str]]) -> pd.DataFrame:
    return pd.DataFrame([{"day": d, "clean_minus_raw_min": o, "status": s} for d, o, s in rows])


class TestMemberNames:
    @pytest.mark.parametrize("house,part,expected", [(1, 1, "RAW_House1_Part1.csv"), (12, 2, "RAW_House12_Part2.csv"),
                                                     (13, 1, "RAW_House_13_Part1.csv"), (16, 2, "RAW_House_16_Part2.csv")])
    def test_raw_names_change_at_house_13(self, house, part, expected):
        assert clock.raw_member_name(house, part) == expected


class TestDailyOffsets:
    def test_finds_official_one_hour_behind_raw(self):
        raw = noise_series("2014-05-01", 4)
        table = clock.daily_offsets(shifted(raw, -60), raw)
        full = table[table["clean_minutes"] == 1440]
        assert (full["clean_minus_raw_min"] == -60).all()
        assert (full["status"] == "verified").all()

    def test_finds_no_offset(self):
        raw = noise_series("2014-05-01", 3)
        assert (clock.daily_offsets(raw, raw)["clean_minus_raw_min"] == 0).all()

    def test_frozen_aggregate_cannot_be_verified(self):
        raw = noise_series("2014-03-01", 2)
        frozen = pd.Series(2169.0, index=raw.index)
        table = clock.daily_offsets(frozen, frozen)
        assert (table["status"] == "uncertain").all()
        assert table["clean_minus_raw_min"].isna().all()
        assert clock.daily_offsets(frozen, raw)["clean_minus_raw_min"].isna().all()


class TestClassifyDays:
    def test_weak_day_between_agreeing_neighbours_is_inferred(self):
        table = clock.classify_days(daily_table([("2014-05-01", -60, "verified"), ("2014-05-02", np.nan, "uncertain"),
                                                 ("2014-05-03", -60, "verified")]))
        assert table["status"].tolist() == ["verified", "inferred", "verified"]
        assert table["clean_minus_raw_min"].iloc[1] == -60

    def test_days_either_side_of_a_change_are_transition(self):
        table = clock.classify_days(daily_table([("2014-03-28", 0, "verified"), ("2014-03-29", 0, "verified"),
                                                 ("2014-03-30", -60, "verified"), ("2014-03-31", -60, "verified")]))
        assert table["status"].tolist() == ["verified", "transition", "transition", "verified"]

    def test_weak_day_at_a_change_stays_uncertain(self):
        table = clock.classify_days(daily_table([("2014-09-03", 0, "verified"), ("2014-09-04", np.nan, "uncertain"),
                                                 ("2014-09-05", -60, "verified"), ("2014-09-06", -60, "verified")]))
        assert table["status"].tolist() == ["transition", "uncertain", "transition", "verified"]
        assert np.isnan(table["clean_minus_raw_min"].iloc[1])


class TestBoundaries:
    def test_periods_are_compressed_and_uncertain_days_merge(self):
        table = daily_table([("2014-03-01", np.nan, "uncertain"), ("2014-03-02", np.nan, "uncertain"),
                             ("2014-03-03", 0, "verified"), ("2014-03-04", 0, "inferred"), ("2014-03-05", -60, "transition")])
        table["r"] = 0.99
        out = clock.boundaries(table, house=10)
        assert out["status"].tolist() == ["uncertain", "usable", "transition"]
        assert out["days"].tolist() == [2, 2, 1]
        assert out["to_get_uk_local_add_min"].iloc[2] == 60

    def test_zero_offset_is_not_negative_zero(self):
        table = daily_table([("2014-03-03", 0, "verified")])
        table["r"] = 0.99
        assert not str(clock.boundaries(table, house=1)["to_get_uk_local_add_min"].iloc[0]).startswith("-")

    def test_negative_zero_float_is_avoided_too(self):
        table = daily_table([("2014-03-03", 0.0, "verified"), ("2014-03-04", np.nan, "uncertain")])
        table["r"] = 0.99
        assert not str(clock.boundaries(table, house=1)["to_get_uk_local_add_min"].iloc[0]).startswith("-")


class TestToUkLocal:
    TABLE = daily_table([("2014-05-01", -60, "verified"), ("2014-05-02", -60, "transition"), ("2014-05-03", np.nan, "uncertain"),
                         ("2014-11-01", 0, "verified")])

    def test_adds_an_hour_where_official_is_behind(self):
        clean = np.array([stamp("2014-05-01 12:00")])
        assert clock.to_uk_local(clean, self.TABLE)["uk_local_seconds"].iloc[0] == stamp("2014-05-01 13:00")

    def test_no_change_where_official_matches_raw(self):
        clean = np.array([stamp("2014-11-01 12:00")])
        assert clock.to_uk_local(clean, self.TABLE)["uk_local_seconds"].iloc[0] == clean[0]

    def test_uncertain_and_transition_days_get_no_local_time(self):
        clean = np.array([stamp("2014-05-02 12:00"), stamp("2014-05-03 12:00"), stamp("2020-01-01")])
        assert clock.to_uk_local(clean, self.TABLE)["uk_local_seconds"].isna().all()

    def test_transition_days_can_be_included_for_validation(self):
        clean = np.array([stamp("2014-05-02 12:00")])
        out = clock.to_uk_local(clean, self.TABLE, include_transition=True)
        assert out["uk_local_seconds"].iloc[0] == stamp("2014-05-02 13:00")

    def test_input_timestamps_are_never_changed(self):
        clean = np.array([stamp("2014-05-01 12:00")])
        original = clean.copy()
        clock.to_uk_local(clean, self.TABLE)
        assert (clean == original).all()

    def test_aggregate_and_appliance_stay_on_the_same_row(self):
        frame = pd.DataFrame({"Unix": [stamp("2014-05-01 12:00"), stamp("2014-05-01 12:01")], "Aggregate": [1.0, 2.0], "Appliance5": [3.0, 4.0]})
        frame["uk_local_seconds"] = clock.to_uk_local(frame["Unix"].to_numpy(), self.TABLE)["uk_local_seconds"].to_numpy()
        assert frame["Aggregate"].tolist() == [1.0, 2.0]
        assert frame["Appliance5"].tolist() == [3.0, 4.0]
        assert frame["Unix"].iloc[0] == stamp("2014-05-01 12:00")


class TestBlockOffsets:
    def test_finds_the_source_and_offset_of_each_block(self):
        raw = noise_series("2014-05-01", 1)
        out = clock.block_offsets(shifted(raw, -60), raw, raw * 0 + 1, "2014-05-01 04:00", "2014-05-01 08:00")
        assert len(out) == 2
        assert all((b["source"], b["clean_minus_raw_min"]) == ("part1", -60) for b in out)

    def test_empty_block_is_reported_not_guessed(self):
        raw = noise_series("2014-05-01", 1)
        out = clock.block_offsets(raw, raw, raw, "2014-05-05 00:00", "2014-05-05 02:00")
        assert out[0]["source"] is None
        assert out[0]["official_minutes"] == 0


class TestPartDrift:
    def test_measures_a_known_lag(self):
        part1 = noise_series("2014-08-04", 3)
        out = clock.part_drift_by_house(part1, shifted(part1, 5))
        assert out["max_lag_min"] == 5
        assert out["lag_on_2014_08_05"] == 5
        assert out["lag_on_2014_09_10"] is None


class TestValidation:
    def test_corrected_spring_change_matches_raw_pattern(self):
        raw_minutes = np.r_[np.arange(stamp("2014-03-30 00:00") // 60, stamp("2014-03-30 01:00") // 60),
                            np.arange(stamp("2014-03-30 02:00") // 60, stamp("2014-03-30 05:00") // 60)]
        official = np.arange(stamp("2014-03-30 01:00") // 60, stamp("2014-03-30 04:00") // 60)
        table = daily_table([("2014-03-30", -60, "transition")])
        results = clock.validate_transitions(official * 60, raw_minutes, table)
        spring = next(r for r in results if r["day"] == "2014-03-30")
        assert spring["raw_minutes_by_hour"] == [60, 0, 60, 60]
        assert spring["corrected_minutes_by_hour"] == [0, 0, 60, 60]
        assert spring["verdict"] == "pass"

    def test_no_raw_data_is_reported_not_passed(self):
        results = clock.validate_transitions(np.array([stamp("2014-05-01")]), np.array([stamp("2014-05-01") // 60]),
                                             daily_table([("2014-05-01", 0, "verified")]))
        assert all(r["verdict"] == "no_data" for r in results)

    def test_routine_shift_detects_a_one_hour_change(self):
        rng = np.random.default_rng(1)
        def year(start: str, end: str, shift: int) -> pd.Series:
            index = np.arange(stamp(start) // 60, stamp(end) // 60)
            profile = 300 + 200 * np.exp(-(((index - shift) % 1440) - 480) ** 2 / 3000.0)
            return pd.Series(profile + rng.normal(0, 5, len(index)), index=index)
        series = pd.concat([year("2014-04-05", "2014-05-31", 0), year("2015-04-05", "2015-05-31", 60)])
        assert clock.routine_shift(series) == 60

    def test_routine_shift_needs_both_years(self):
        index = np.arange(stamp("2014-04-05") // 60, stamp("2014-05-31") // 60)
        assert clock.routine_shift(pd.Series(1.0, index=index)) is None

    def test_combine_raw_prefers_part1(self):
        part1 = pd.Series([1.0, 2.0], index=[10, 11])
        part2 = pd.Series([9.0, 3.0], index=[11, 12])
        assert clock.combine_raw(part1, part2).to_dict() == {10: 1.0, 11: 2.0, 12: 3.0}
