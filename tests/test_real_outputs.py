"""Invariants on the real pipeline outputs. Skipped until the pipeline has been run."""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import (AGGREGATE_REVIEW_W, APPLIANCE_LIMIT_W, APPLIANCES, CLOCK_DAILY_CSV, CONFLICTS_PARQUET, HOUSE1_PARQUET,
                        HOUSEHOLDS, INPUT_SHA256, LONG_OUTAGE_MINUTES, RAW_ARCHIVE, SENSORS, official_parquet)
from src.io_utils import sha256_file

pytestmark = pytest.mark.skipif(not HOUSE1_PARQUET.exists(), reason="Run python -m src.pipeline first.")


@pytest.fixture(scope="module")
def minutes() -> pd.DataFrame:
    return pd.read_parquet(HOUSE1_PARQUET)


def test_grid_has_one_row_per_minute_with_no_gaps(minutes):
    assert minutes["timestamp_utc"].is_unique
    assert (minutes["timestamp_utc"].diff().dropna() == pd.Timedelta(minutes=1)).all()


def test_raw_clock_is_utc_plus_recorded_offset(minutes):
    naive = minutes["timestamp_utc"].dt.tz_localize(None)
    assert (minutes["raw_clock_minute"] - naive == pd.to_timedelta(minutes["utc_offset_min"], unit="min")).all()


def test_no_appliance_value_exceeds_the_limit(minutes):
    assert not (minutes[APPLIANCES] > APPLIANCE_LIMIT_W).any().any()


def test_aggregate_above_review_level_is_kept(minutes):
    assert (minutes["Aggregate_over_23kw"] > 0).sum() > 0
    assert minutes["Aggregate"].max() > AGGREGATE_REVIEW_W


def test_no_negative_power(minutes):
    assert not (minutes[SENSORS] < 0).any().any()


@pytest.mark.parametrize("sensor", SENSORS)
def test_missing_flag_count_and_value_agree(minutes, sensor):
    assert (minutes[sensor].isna() == minutes[f"{sensor}_missing"]).all()
    assert ((minutes[f"{sensor}_n"] == 0) == minutes[f"{sensor}_missing"]).all()


def test_long_outages_stay_missing(minutes):
    outage = minutes[minutes["long_outage"]]
    assert len(outage) > 0
    assert outage[SENSORS].isna().all().all()
    assert (outage["outage_minutes"] >= LONG_OUTAGE_MINUTES).all()


def test_part_switch_matches_the_measured_boundary(minutes):
    switch = minutes.loc[minutes["source_part"] == 2, "raw_clock_minute"].iloc[0]
    assert switch == pd.Timestamp("2014-06-27 21:57")
    assert (minutes.loc[minutes["raw_clock_minute"] > switch, "source_part"].dropna() == 2).all()


def test_part1_zeros_are_preserved(minutes):
    part1 = minutes[minutes["source_part"] == 1]
    assert (part1["Appliance5"] == 0).mean() > 0.9


def test_each_clock_change_is_handled(minutes):
    spring = minutes[(minutes["timestamp_utc"] >= "2014-03-30 00:00:00+00:00") & (minutes["timestamp_utc"] < "2014-03-30 03:00:00+00:00")]
    assert not spring["Aggregate_missing"].any()
    assert not (spring["raw_clock_minute"].dt.hour == 1).any()
    autumn_2013 = minutes[(minutes["timestamp_utc"] >= "2013-10-27 00:00:00+00:00") & (minutes["timestamp_utc"] < "2013-10-27 02:00:00+00:00")]
    assert not autumn_2013["dst_ambiguous"].any()
    assert (autumn_2013["raw_clock_minute"].dt.hour == 1).sum() == 120
    assert minutes["dst_ambiguous"].sum() == 120


def test_flagged_frozen_values_are_kept(minutes):
    frozen = minutes[minutes["Appliance5_frozen"]]
    assert len(frozen) > 0
    assert frozen["Appliance5"].notna().all()


def test_conflict_log_resolves_to_the_table():
    log = pd.read_parquet(CONFLICTS_PARQUET)
    assert len(log) > 0
    assert log["sensor"].isin(SENSORS).all()
    within = log.dropna(subset=["resolved_w"])
    assert ((within["resolved_w"] >= within["low"] - 1e-9) & (within["resolved_w"] <= within["high"] + 1e-9)).all()


def test_clock_table_covers_every_household():
    daily = pd.read_csv(CLOCK_DAILY_CSV)
    assert set(daily["house"]) == set(HOUSEHOLDS)
    assert daily["status"].isin(["verified", "inferred", "transition", "uncertain"]).all()
    usable = daily[daily["status"].isin(["verified", "inferred"])]
    assert usable["clean_minus_raw_min"].isin([-60, 0]).all()


def test_archives_still_match_recorded_checksums():
    assert sha256_file(RAW_ARCHIVE) == INPUT_SHA256[RAW_ARCHIVE]


OFFICIAL_READY = all(official_parquet(house).exists() for house in HOUSEHOLDS)


@pytest.mark.skipif(not OFFICIAL_READY, reason="Run python -m src.pipeline first.")
@pytest.mark.parametrize("house", list(HOUSEHOLDS))
def test_prepared_tables_are_consistent(house):
    table = pd.read_parquet(official_parquet(house))
    assert (table["official_minute"].diff().dropna() == pd.Timedelta(minutes=1)).all()
    assert (table["valid"] == (~table["missing"] & ~table["frozen"])).all()
    assert table.loc[table["missing"], "aggregate_w"].isna().all()
    assert table["frozen"].sum() == (table["aggregate_frozen"] | table["washing_machine_frozen"]).sum()


@pytest.mark.skipif(not OFFICIAL_READY, reason="Run python -m src.pipeline first.")
def test_clock_correction_never_changes_original_minutes():
    table = pd.read_parquet(official_parquet(1), columns=["official_minute", "uk_local_minute"]).dropna()
    difference = (table["uk_local_minute"] - table["official_minute"]).dt.total_seconds() / 60
    assert set(difference.unique()) <= {0.0, 60.0}
