"""Measure raw and cleaned House 1 for the cleaning report. Nothing here changes the data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.cleaning.raw_checks import backward_steps, duplicate_overview, stamp
from src.cleaning.raw_house1 import CleanResult, spike_mask
from src.config import (AGGREGATE_REVIEW_W, APPLIANCE_LIMIT_W, APPLIANCES, DST_TRANSITIONS, HOUSE1_LABELS,
                        LONG_OUTAGE_MINUTES, SENSORS)

# Exact 2^k and 2^k - 1 values, the signature of a sensor bit error.
BIT_PATTERNS = [2**k for k in range(12, 18)] + [2**k - 1 for k in range(13, 18)]
OUTAGE_BUCKETS = [(1, 3, "1 to 2 min"), (3, 10, "3 to 9 min"), (10, 60, "10 to 59 min"), (60, 1440, "1 to 24 h"),
                  (1440, 10**9, "over 24 h")]


def part_profile(df: pd.DataFrame) -> dict:
    """Format and quality facts for one raw file, exactly as supplied."""
    unix = df["Unix"].to_numpy()
    gaps = np.diff(np.sort(unix))
    per_sensor = []
    for sensor in SENSORS:
        values = df[sensor]
        per_sensor.append({
            "sensor": sensor,
            "label": HOUSE1_LABELS[sensor],
            "missing_rows": int(values.isna().sum()),
            "missing_pct": round(float(values.isna().mean() * 100), 2),
            "zero_pct": round(float((values == 0).mean() * 100), 2),
            "max_w": float(values.max()),
        })
    return {
        "first_time": str(df["Time"].iloc[0]),
        "last_time": str(df["Time"].iloc[-1]),
        "backward_steps": backward_steps(df),
        "gaps_over_2min": int((gaps > 120).sum()),
        "days_in_gaps_over_2min": round(float(gaps[gaps > 120].sum() / 86400), 1),
        "gaps_over_1h": int((gaps > 3600).sum()),
        "gaps_over_1d": int((gaps > 86400).sum()),
        "days_in_gaps_over_1d": round(float(gaps[gaps > 86400].sum() / 86400), 1),
        "median_positive_gap_s": float(np.median(gaps[gaps > 0])),
        **duplicate_overview(df),
        "sensors": per_sensor,
    }


def missing_profile(minutes: pd.DataFrame) -> list[dict]:
    """Coverage and energy per sensor in the cleaned table."""
    total = len(minutes)
    rows = []
    for sensor in SENSORS:
        missing = minutes[f"{sensor}_missing"]
        present = minutes.loc[~missing, sensor]
        rows.append({
            "sensor": sensor,
            "label": HOUSE1_LABELS[sensor],
            "minutes_present": int((~missing).sum()),
            "minutes_missing": int(missing.sum()),
            "missing_pct": round(float(missing.mean() * 100), 2),
            "mean_w_when_present": round(float(present.mean()), 2),
            "energy_kwh_measured": round(float(present.sum() / 60000), 1),
            "median_readings_per_minute": float(minutes.loc[~missing, f"{sensor}_n"].median()),
            "minutes_with_under_3_readings": int(((minutes[f"{sensor}_n"] > 0) & (minutes[f"{sensor}_n"] < 3)).sum()),
        })
    return rows + [{"sensor": "grid_total_minutes", "minutes_present": total}]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype("int8"), 0]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def outage_profile(minutes: pd.DataFrame) -> dict:
    """Whole-house outages and per-sensor silences in the cleaned table."""
    is_outage = minutes["outage_minutes"].to_numpy() > 0
    runs = _runs(is_outage)
    lengths = np.array([b - a for a, b in runs]) if runs else np.array([], dtype=int)
    longest = sorted(runs, key=lambda r: r[0] - r[1])[:10]
    buckets = [{
        "length": label,
        "outages": int(((lengths >= lo) & (lengths < hi)).sum()),
        "total_days": round(float(lengths[(lengths >= lo) & (lengths < hi)].sum() / 1440), 2),
    } for lo, hi, label in OUTAGE_BUCKETS]
    sensors = []
    for sensor in SENSORS:
        sensor_runs = _runs(minutes[f"{sensor}_gap_minutes"].to_numpy() >= LONG_OUTAGE_MINUTES)
        sensor_lengths = [b - a for a, b in sensor_runs]
        only_sensor = minutes[f"{sensor}_missing"].to_numpy() & ~is_outage & ~minutes["dst_ambiguous"].to_numpy()
        sensors.append({
            "sensor": sensor,
            "long_gaps": len(sensor_runs),
            "total_days_in_long_gaps": round(sum(sensor_lengths) / 1440, 2),
            "longest_gap_days": round(max(sensor_lengths, default=0) / 1440, 2),
            "missing_minutes_while_other_sensors_report": int(only_sensor.sum()),
        })
    return {
        "whole_house_outage_minutes": int(is_outage.sum()),
        "whole_house_outage_days": round(float(is_outage.sum() / 1440), 2),
        "outages_of_any_length": len(runs),
        "short_outages_under_threshold": int((lengths < LONG_OUTAGE_MINUTES).sum()),
        "long_outage_threshold_minutes": LONG_OUTAGE_MINUTES,
        "long_outages": int((lengths >= LONG_OUTAGE_MINUTES).sum()),
        "buckets": buckets,
        "longest": [{
            "raw_clock_start": str(minutes["raw_clock_minute"].iloc[a]),
            "days": round((b - a) / 1440, 2),
            "minutes": b - a,
        } for a, b in longest],
        "per_sensor": sensors,
    }


def spike_profile(rows: pd.DataFrame, minutes: pd.DataFrame) -> list[dict]:
    """Appliance readings above the limit, which became missing."""
    spikes = spike_mask(rows)
    out = []
    for sensor in APPLIANCES:
        values = rows.loc[spikes[sensor], sensor]
        counted = minutes[f"{sensor}_spikes"].to_numpy()
        out.append({
            "sensor": sensor,
            "label": HOUSE1_LABELS[sensor],
            "readings_over_limit": len(values),
            "part1_readings": int((spikes[sensor] & (rows["part"] == 1)).sum()),
            "part2_readings": int((spikes[sensor] & (rows["part"] == 2)).sum()),
            "bit_pattern_pct": round(float(values.isin(BIT_PATTERNS).mean() * 100), 1) if len(values) else 0.0,
            "commonest_values": {str(k): int(v) for k, v in values.value_counts().head(5).items()},
            "max_w": float(values.max()) if len(values) else None,
            "minutes_affected": int((counted > 0).sum()),
            "minutes_left_with_no_reading": int(((counted > 0) & minutes[f"{sensor}_missing"].to_numpy()).sum()),
        })
    return out


def aggregate_review(rows: pd.DataFrame, minutes: pd.DataFrame) -> dict:
    """Investigate aggregate readings above the review threshold. Nothing is removed."""
    high = rows.sort_values("utc", kind="stable").reset_index(drop=True)
    values = high["Aggregate"].to_numpy()
    over = np.flatnonzero(values > AGGREGATE_REVIEW_W)
    valid = np.flatnonzero(~np.isnan(values))
    episodes = []
    if len(over):
        for block in np.split(over, np.flatnonzero(np.diff(high["utc"].to_numpy()[over]) > 600) + 1):
            first, last = block[0], block[-1]
            before = values[valid[valid < first][-1]] if (valid < first).any() else np.nan
            after = values[valid[valid > last][0]] if (valid > last).any() else np.nan
            isolated = len(block) == 1 and np.nanmax([before, after]) < 5000
            episodes.append({
                "raw_clock_start": str(pd.Timestamp(int(high["Unix"].iloc[first]), unit="s")),
                "rows": len(block),
                "max_w": float(values[block].max()),
                "reading_before_w": float(before),
                "reading_after_w": float(after),
                "appliance_sum_w": float(high.loc[first, APPLIANCES].sum()),
                "kind": "isolated single reading" if isolated else "sustained high load",
            })
    flagged = minutes[minutes["Aggregate_over_23kw"] > 0]
    minute_key = high["utc"].to_numpy() // 60
    keep_minutes = np.isin(minute_key, minute_key[over])
    subset = pd.DataFrame({"m": minute_key[keep_minutes], "w": values[keep_minutes]})
    with_all = subset.groupby("m")["w"].mean()
    without = subset[~(subset["w"] > AGGREGATE_REVIEW_W)].groupby("m")["w"].mean().reindex(with_all.index)
    effect = (with_all - without).dropna()
    thresholds = {f"over_{w}_w": int((values > w).sum()) for w in (8000, 10000, 12000, 15000, 20000, AGGREGATE_REVIEW_W)}
    return {
        "threshold_w": AGGREGATE_REVIEW_W,
        "readings_over_threshold": len(over),
        "minutes_flagged": len(flagged),
        "readings_at_exactly_24000_w": int((values == 24000).sum()),
        "readings_by_threshold": thresholds,
        "episodes": episodes,
        "isolated_readings": sum(e["kind"] == "isolated single reading" for e in episodes),
        "effect_on_minute_mean_w": {
            "median": round(float(effect.median()), 1) if len(effect) else 0.0,
            "max": round(float(effect.max()), 1) if len(effect) else 0.0,
        },
    }


def frozen_profile(minutes: pd.DataFrame) -> dict:
    """Sensors stuck on one exact value. Values stay in the table and are flagged."""
    per_sensor, episodes = [], []
    for sensor in SENSORS:
        mask = minutes[f"{sensor}_frozen"].to_numpy()
        runs = _runs(mask)
        present = minutes.loc[~minutes[f"{sensor}_missing"], sensor]
        frozen_kwh = float(minutes.loc[mask, sensor].sum() / 60000)
        total_kwh = float(present.sum() / 60000)
        per_sensor.append({
            "sensor": sensor,
            "label": HOUSE1_LABELS[sensor],
            "runs": len(runs),
            "minutes": int(mask.sum()),
            "frozen_energy_kwh": round(frozen_kwh, 2),
            "share_of_sensor_energy_pct": round(frozen_kwh / total_kwh * 100, 2) if total_kwh else 0.0,
        })
        for a, b in runs:
            episodes.append({"sensor": sensor, "raw_clock_start": str(minutes["raw_clock_minute"].iloc[a]), "minutes": b - a,
                             "value_w": round(float(minutes[sensor].iloc[a]), 1), "source_part": int(minutes["source_part"].iloc[a])})
    episodes.sort(key=lambda e: -e["minutes"])
    return {"per_sensor": per_sensor, "longest": episodes[:12]}


def conflict_profile(conflicts: pd.DataFrame) -> list[dict]:
    out = []
    for sensor in SENSORS:
        part = conflicts[conflicts["sensor"] == sensor]
        out.append({
            "sensor": sensor,
            "label": HOUSE1_LABELS[sensor],
            "conflicts": len(part),
            "zero_vs_nonzero": int((part["kind"] == "zero_vs_nonzero").sum()),
            "both_nonzero": int((part["kind"] == "both_nonzero").sum()),
            "involves_spike": int(part["involves_spike"].sum()),
            "spread_100w_or_more": int((part["spread_w"] >= 100).sum()),
            "median_spread_w": float(part["spread_w"].median()) if len(part) else 0.0,
            "max_spread_w": float(part["spread_w"].max()) if len(part) else 0.0,
        })
    return out


def part1_zero_profile(minutes: pd.DataFrame) -> list[dict]:
    """Minutes from Part 1 reading exactly zero, where off and unavailable look identical."""
    part1 = minutes[minutes["source_part"] == 1]
    return [{
        "sensor": sensor,
        "minutes_with_reading": int((~part1[f"{sensor}_missing"]).sum()),
        "minutes_exactly_zero": int((part1[sensor] == 0).sum()),
        "zero_pct": round(float((part1.loc[~part1[f"{sensor}_missing"], sensor] == 0).mean() * 100), 1),
    } for sensor in SENSORS]


def dst_profile(minutes: pd.DataFrame) -> list[dict]:
    """Minutes with a reading in each UTC hour around each UK clock change."""
    utc = (minutes["timestamp_utc"] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    utc = utc.to_numpy()
    has_reading = ~minutes["Aggregate_missing"].to_numpy()
    readings = minutes["Aggregate_n"].to_numpy()
    out = []
    for day, direction in DST_TRANSITIONS.items():
        start = stamp(day)
        hours = []
        for hour in range(-1, 4):
            lo = start + hour * 3600
            here = (utc >= lo) & (utc < lo + 3600)
            hours.append({"utc_hour": str(pd.Timestamp(lo, unit="s"))[5:16], "minutes_with_reading": int(has_reading[here].sum()),
                          "readings": int(readings[here].sum()), "dst_ambiguous_minutes": int(minutes["dst_ambiguous"].to_numpy()[here].sum())})
        out.append({"day": day, "direction": direction, "hours": hours})
    return out


def build_stats(part1: pd.DataFrame, part2: pd.DataFrame, result: CleanResult) -> dict:
    """Everything the cleaning report needs, as JSON-ready values."""
    minutes = result.minutes
    return {
        "row_flow": result.flow,
        "raw_files": {"part1": part_profile(part1), "part2": part_profile(part2)},
        "missing_values": missing_profile(minutes),
        "duplicates": {
            "exact_duplicates_removed": result.flow["exact_duplicate_rows_removed"],
            "rows_merged_into_an_existing_timestamp": result.flow["rows_merged_into_an_existing_timestamp"],
            "conflicts_logged": len(result.conflicts),
            "timestamps_with_a_conflict": int(result.conflicts["timestamp_utc"].nunique()),
            "minutes_with_a_conflict": int((minutes["n_conflicts"] > 0).sum()),
            "per_sensor": conflict_profile(result.conflicts),
        },
        "outliers": {
            "appliance_limit_w": APPLIANCE_LIMIT_W,
            "appliance_spikes": spike_profile(result.rows, minutes),
            "aggregate_review": aggregate_review(result.rows, minutes),
        },
        "outages": outage_profile(minutes),
        "frozen": frozen_profile(minutes),
        "part1_zeros": part1_zero_profile(minutes),
        "clock": {
            "dst_transitions": dst_profile(minutes),
            "dst_ambiguous_minutes": int(minutes["dst_ambiguous"].sum()),
            "clock_uncertain_minutes": int(minutes["clock_uncertain"].sum()),
            "minutes_from_part1": int((minutes["source_part"] == 1).sum()),
            "minutes_from_part2": int((minutes["source_part"] == 2).sum()),
        },
    }
