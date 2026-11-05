"""Clean raw House 1 into a 1-minute table. DATA.md records the evidence for each rule.

The functions are pure: they take data frames and return new ones. No reading is
interpolated or filled, and original timestamps are carried into the output.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.cleaning.raw_checks import find_conflicts, stamp
from src.config import (AGGREGATE_REVIEW_W, APPLIANCE_LIMIT_W, APPLIANCES, CLOCK_UNCERTAIN_RAW, DST_TRANSITIONS,
                        FROZEN_MIN_APPLIANCE_W, FROZEN_MIN_MINUTES, LONG_OUTAGE_MINUTES, SENSORS, TIMEZONE)

_EPOCH_UTC = pd.Timestamp("1970-01-01", tz="UTC")


@dataclass
class CleanResult:
    minutes: pd.DataFrame
    conflicts: pd.DataFrame
    rows: pd.DataFrame  # Rows used, after source selection and exact-duplicate removal.
    flow: dict


# --------------------------------------------------------------------------- clock
def assign_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Add true UTC seconds to each row, leaving the original timestamp columns untouched.

    Raw timestamps are UK local clock time. In the repeated autumn hour, a backward step in
    file order separates the two passes. Without one the passes are interleaved, so all
    rows go to the first pass and are marked dst_ambiguous.
    """
    out = df.copy()
    unix = out["Unix"].to_numpy()
    is_summer_time = np.ones(len(out), dtype=bool)
    ambiguous = np.zeros(len(out), dtype=bool)
    for day, direction in DST_TRANSITIONS.items():
        if direction != "back":
            continue
        start = stamp(f"{day} 01:00")
        where = np.flatnonzero((unix >= start) & (unix < start + 3600))
        if len(where) == 0:
            continue
        steps = np.flatnonzero(np.diff(unix[where]) < 0)
        if len(steps) == 1:
            is_summer_time[where[steps[0] + 1:]] = False
        else:
            ambiguous[where] = True
    local = pd.to_datetime(unix, unit="s")
    aware = local.tz_localize(TIMEZONE, ambiguous=is_summer_time, nonexistent="raise")
    out["utc"] = ((aware - _EPOCH_UTC) // pd.Timedelta(seconds=1)).to_numpy().astype("int64")
    out["dst_ambiguous"] = ambiguous
    return out


def ambiguous_utc_windows(rows: pd.DataFrame) -> list[tuple[int, int]]:
    """UTC spans of autumn clock hours whose two passes could not be separated."""
    windows = []
    if not rows["dst_ambiguous"].any():
        return windows
    utc = rows.loc[rows["dst_ambiguous"], "utc"].to_numpy()
    for day, direction in DST_TRANSITIONS.items():
        start = stamp(day)
        if direction == "back" and ((utc >= start) & (utc < start + 7200)).any():
            windows.append((start, start + 7200))
    return windows


# --------------------------------------------------------------------------- row rules
def select_source(part1: pd.DataFrame, part2: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Use Part 1 strictly before the first Part 2 timestamp and Part 2 from there."""
    switch_utc = int(part2["utc"].min())
    combined = pd.concat([part1[part1["utc"] < switch_utc], part2], ignore_index=True)
    return combined, switch_utc


def drop_exact_duplicates(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep one row where the timestamp and all ten readings are identical."""
    return rows[~rows.duplicated(subset=["utc", *SENSORS], keep="first")].reset_index(drop=True)


def spike_mask(rows: pd.DataFrame) -> pd.DataFrame:
    """True where an appliance reading exceeds the rated monitor limit."""
    return rows[APPLIANCES] > APPLIANCE_LIMIT_W


def merge_timestamps(rows: pd.DataFrame) -> pd.DataFrame:
    """Reduce rows to one per timestamp, sensor by sensor, averaging all non-null readings.

    Spikes become missing first, so they cannot influence a merged value. A NaN in one row
    never removes another row's reading. Spike and review counts are kept per timestamp.
    """
    readings = rows[SENSORS].copy()
    spikes = spike_mask(rows)
    readings[APPLIANCES] = readings[APPLIANCES].where(~spikes)
    groups = rows["utc"].to_numpy()
    merged = readings.groupby(groups, sort=True).mean()
    for sensor in APPLIANCES:
        merged[f"{sensor}_spikes"] = spikes[sensor].groupby(groups, sort=True).sum().astype("int32")
    merged["Aggregate_over_review"] = (rows["Aggregate"] > AGGREGATE_REVIEW_W).groupby(groups, sort=True).sum().astype("int32")
    merged["source_part"] = rows["part"].groupby(groups, sort=True).max()
    merged["dst_ambiguous"] = rows["dst_ambiguous"].groupby(groups, sort=True).any()
    merged.index.name = "utc"
    return merged


def log_conflicts(rows: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """Record every conflict with the value it resolved to, so none is silently discarded."""
    conflicts = find_conflicts(rows, key="utc")
    conflicts["resolved_w"] = [merged.at[utc, sensor] for utc, sensor in zip(conflicts["utc"], conflicts["sensor"])]
    conflicts["raw_clock"] = rows.drop_duplicates("utc").set_index("utc")["Unix"].reindex(conflicts["utc"]).to_numpy()
    conflicts["raw_clock"] = pd.to_datetime(conflicts["raw_clock"], unit="s")
    conflicts["timestamp_utc"] = pd.to_datetime(conflicts["utc"], unit="s", utc=True)
    return conflicts.drop(columns="utc")


# --------------------------------------------------------------------------- minute table
def run_lengths(mask: np.ndarray) -> np.ndarray:
    """For each True element, the length of the run of True values containing it."""
    result = np.zeros(len(mask), dtype="int32")
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype("int8"), 0]))
    for start, stop in zip(edges[::2], edges[1::2]):
        result[start:stop] = stop - start
    return result


def frozen_runs(minute_index: np.ndarray, values: np.ndarray, min_minutes: int = FROZEN_MIN_MINUTES,
                min_value: float | None = None) -> list[tuple[int, int, float]]:
    """Find runs of one exact value across consecutive minutes, as (start position, length, value).

    Real household power fluctuates, so an identical minute mean held this long means a
    stuck sensor or logger. Missing minutes break a run.
    """
    v = np.asarray(values, dtype="float64")
    same = (v[1:] == v[:-1]) & (np.diff(minute_index) == 1)
    edges = np.flatnonzero(np.diff(np.r_[0, same.astype("int8"), 0]))
    runs = []
    for start, stop in zip(edges[::2].tolist(), edges[1::2].tolist()):
        length = stop - start + 1
        if length >= min_minutes and (min_value is None or v[start] >= min_value):
            runs.append((start, length, float(v[start])))
    return runs


def frozen_mask(minute_grid: np.ndarray, values: np.ndarray, min_value: float | None) -> np.ndarray:
    """True for minutes inside a frozen run. Only for flagging: values are never changed."""
    present = ~np.isnan(values)
    mask = np.zeros(len(values), dtype=bool)
    where = np.flatnonzero(present)
    if len(where) == 0:
        return mask
    for start, length, _ in frozen_runs(minute_grid[where], values[where], min_value=min_value):
        mask[where[start:start + length]] = True
    return mask


def to_minutes(merged: pd.DataFrame, conflicts: pd.DataFrame, ambiguous_windows: list[tuple[int, int]]) -> pd.DataFrame:
    """Average available readings per UTC minute on a complete grid, without filling anything."""
    minute = merged.index.to_numpy() // 60
    grid = np.arange(minute.min(), minute.max() + 1)
    means = merged[SENSORS].groupby(minute).mean().reindex(grid)
    counts = merged[SENSORS].notna().groupby(minute).sum().reindex(grid, fill_value=0)

    stamps = pd.to_datetime(grid * 60, unit="s", utc=True)
    local = stamps.tz_convert(TIMEZONE).tz_localize(None)
    columns: dict[str, np.ndarray | pd.Series] = {
        "timestamp_utc": stamps,
        "raw_clock_minute": local,
        "utc_offset_min": ((local - stamps.tz_localize(None)) // pd.Timedelta(minutes=1)).to_numpy().astype("int8"),
        "source_part": merged["source_part"].groupby(minute).max().reindex(grid).astype("Int8").array,
    }
    frame = pd.DataFrame(columns)

    seconds = grid * 60
    in_window = np.zeros(len(grid), dtype=bool)
    for start, stop in ambiguous_windows:
        in_window |= (seconds >= start) & (seconds < stop)
    frame["dst_ambiguous"] = in_window
    lo, hi = stamp(CLOCK_UNCERTAIN_RAW[0]), stamp(CLOCK_UNCERTAIN_RAW[1])
    raw_seconds = (local - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    frame["clock_uncertain"] = (raw_seconds >= lo) & (raw_seconds < hi)

    sensor_columns: dict[str, np.ndarray] = {}
    for sensor in SENSORS:
        n = counts[sensor].to_numpy()
        missing = n == 0
        sensor_columns[sensor] = means[sensor].to_numpy().astype("float32")
        sensor_columns[f"{sensor}_n"] = n.astype("int16")
        sensor_columns[f"{sensor}_missing"] = missing
        sensor_columns[f"{sensor}_gap_minutes"] = run_lengths(missing)
        sensor_columns[f"{sensor}_frozen"] = frozen_mask(
            grid, means[sensor].to_numpy(), None if sensor == "Aggregate" else FROZEN_MIN_APPLIANCE_W)
    for sensor in APPLIANCES:
        sensor_columns[f"{sensor}_spikes"] = (
            merged[f"{sensor}_spikes"].groupby(minute).sum().reindex(grid, fill_value=0).to_numpy().astype("int16"))
    sensor_columns["Aggregate_over_23kw"] = (
        merged["Aggregate_over_review"].groupby(minute).sum().reindex(grid, fill_value=0).to_numpy().astype("int16"))
    frame = pd.concat([frame, pd.DataFrame(sensor_columns)], axis=1)

    if len(conflicts):
        per_minute = ((conflicts["timestamp_utc"] - _EPOCH_UTC) // pd.Timedelta(minutes=1)).value_counts()
        frame["n_conflicts"] = per_minute.reindex(grid, fill_value=0).to_numpy().astype("int16")
    else:
        frame["n_conflicts"] = np.zeros(len(grid), dtype="int16")

    no_reading = frame[[f"{s}_missing" for s in SENSORS]].all(axis=1).to_numpy()
    outage = no_reading & ~in_window
    frame["outage_minutes"] = run_lengths(outage)
    frame["long_outage"] = frame["outage_minutes"].to_numpy() >= LONG_OUTAGE_MINUTES
    return frame


# --------------------------------------------------------------------------- orchestration
def clean_house1(part1: pd.DataFrame, part2: pd.DataFrame) -> CleanResult:
    """Apply every cleaning rule in order and return the table, conflict log and row counts."""
    p1, p2 = assign_utc(part1), assign_utc(part2)
    combined, switch_utc = select_source(p1, p2)
    rows = drop_exact_duplicates(combined)
    merged = merge_timestamps(rows)
    conflicts = log_conflicts(rows, merged)
    minutes = to_minutes(merged, conflicts, ambiguous_utc_windows(rows))
    flow = {
        "part1_rows_read": len(part1),
        "part2_rows_read": len(part2),
        "switch_timestamp_utc": str(pd.Timestamp(switch_utc, unit="s", tz="UTC")),
        "part1_rows_kept_before_switch": int((combined["part"] == 1).sum()),
        "part1_rows_not_used_after_switch": int(len(part1) - (combined["part"] == 1).sum()),
        "part2_rows_kept": int((combined["part"] == 2).sum()),
        "rows_after_source_selection": len(combined),
        "exact_duplicate_rows_removed": len(combined) - len(rows),
        "exact_duplicates_removed_part1": int((combined["part"] == 1).sum() - (rows["part"] == 1).sum()),
        "exact_duplicates_removed_part2": int((combined["part"] == 2).sum() - (rows["part"] == 2).sum()),
        "rows_after_exact_duplicates": len(rows),
        "unique_timestamps": len(merged),
        "rows_merged_into_an_existing_timestamp": len(rows) - len(merged),
        "minutes_in_grid": len(minutes),
        "minutes_with_any_reading": int((~minutes[[f"{s}_missing" for s in SENSORS]].all(axis=1)).sum()),
    }
    return CleanResult(minutes=minutes, conflicts=conflicts, rows=rows, flow=flow)
