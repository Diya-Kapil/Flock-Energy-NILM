"""Pre-cleaning checks on raw House 1: duplicate conflicts, the part boundary, and clock changes.

These functions only measure. Nothing here changes or removes a reading.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import APPLIANCE_LIMIT_W, APPLIANCES, DST_TRANSITIONS, SENSORS


def stamp(day: str) -> int:
    """Raw clock label for a date or datetime string, as stored in the Unix column."""
    return int(pd.Timestamp(day).value // 10**9)


# --------------------------------------------------------------------------- duplicates
def duplicate_overview(df: pd.DataFrame) -> dict:
    """Count duplicate timestamps and how they relate to each other."""
    columns = ["Unix", *SENSORS]
    repeated = df["Unix"].duplicated(keep=False)
    return {
        "rows": len(df),
        "unique_timestamps": int(df["Unix"].nunique()),
        "rows_on_a_repeated_timestamp": int(repeated.sum()),
        "extra_rows_beyond_first": int(df["Unix"].duplicated().sum()),
        "rows_identical_to_an_earlier_row": int(df.duplicated(subset=columns).sum()),
        "time_and_unix_disagree_rows": int((df["Time"].astype("int64") != df["Unix"]).sum()),
        "max_rows_on_one_timestamp": int(df.groupby("Unix").size().max()),
    }


def find_conflicts(df: pd.DataFrame, limit: float | None = None, key: str = "Unix") -> pd.DataFrame:
    """List every sensor and timestamp where rows hold different non-null readings.

    Returns one row per conflict with the first and last readings in file order. With
    `limit` set, appliance readings above it are ignored, as spike removal would do.
    """
    subset = df.loc[df[key].duplicated(keep=False), [key, *SENSORS]].copy()
    if limit is not None:
        for sensor in APPLIANCES:
            subset[sensor] = subset[sensor].where(subset[sensor] <= limit)
    found = []
    for sensor in SENSORS:
        table = subset.groupby(key, sort=True)[sensor].agg(
            n_readings="count", n_distinct="nunique", low="min", high="max", first_value="first", last_value="last",
            mean_value="mean")
        table = table[table["n_distinct"] > 1].reset_index()
        table.insert(1, "sensor", sensor)
        found.append(table)
    conflicts = pd.concat(found, ignore_index=True)
    conflicts["kind"] = np.where(conflicts["low"] == 0, "zero_vs_nonzero", "both_nonzero")
    limit_w = APPLIANCE_LIMIT_W if limit is None else limit
    conflicts["involves_spike"] = conflicts["sensor"].isin(APPLIANCES) & (conflicts["high"] > limit_w)
    conflicts["spread_w"] = conflicts["high"] - conflicts["low"]
    return conflicts


def conflict_summary(conflicts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sensor in SENSORS:
        part = conflicts[conflicts["sensor"] == sensor]
        rows.append({
            "sensor": sensor,
            "conflicts": len(part),
            "zero_vs_nonzero": int((part["kind"] == "zero_vs_nonzero").sum()),
            "both_nonzero": int((part["kind"] == "both_nonzero").sum()),
            "involves_spike": int(part["involves_spike"].sum()),
            "median_spread_w": float(part["spread_w"].median()) if len(part) else 0.0,
            "p99_spread_w": float(part["spread_w"].quantile(0.99)) if len(part) else 0.0,
            "max_spread_w": float(part["spread_w"].max()) if len(part) else 0.0,
        })
    return pd.DataFrame(rows)


def neighbour_evidence(df: pd.DataFrame, sensor: str) -> dict:
    """Test which conflicting reading fits the readings either side of it.

    If rows are written in time order, the last reading at a timestamp should match the
    next timestamp more often than the first one does.
    """
    table = df.dropna(subset=[sensor]).groupby("Unix", sort=True)[sensor].agg(["first", "last"])
    following = table["first"].shift(-1)
    preceding = table["last"].shift(1)
    conflict = table["first"] != table["last"]
    n = int(conflict.sum())
    if n == 0:
        return {"sensor": sensor, "conflicts": 0}
    return {
        "sensor": sensor,
        "conflicts": n,
        "last_equals_next_pct": round(float((table["last"] == following)[conflict].mean() * 100), 1),
        "first_equals_next_pct": round(float((table["first"] == following)[conflict].mean() * 100), 1),
        "first_equals_previous_pct": round(float((table["first"] == preceding)[conflict].mean() * 100), 1),
        "last_equals_previous_pct": round(float((table["last"] == preceding)[conflict].mean() * 100), 1),
    }


def rule_impact(df: pd.DataFrame, sensor: str, limit: float | None = APPLIANCE_LIMIT_W) -> dict:
    """Compare candidate merge rules by their effect on the 1-minute mean."""
    values = df.loc[df[sensor].notna(), ["Unix", sensor]].copy()
    if limit is not None and sensor in APPLIANCES:
        values = values[values[sensor] <= limit]
    by_time = values.groupby("Unix", sort=True)[sensor]
    rules = pd.DataFrame({"first": by_time.first(), "last": by_time.last(), "mean": by_time.mean()})
    minute = rules.index // 60
    means = rules.groupby(minute).mean()
    affected = rules.loc[rules["first"] != rules["last"]].index // 60
    affected = means.index.isin(np.unique(affected))
    gap = (means["last"] - means["mean"]).abs()
    gap_first = (means["first"] - means["last"]).abs()
    return {
        "sensor": sensor,
        "minutes": len(means),
        "minutes_with_a_conflict": int(affected.sum()),
        "last_vs_mean_abs_w_p50_in_affected": round(float(gap[affected].median()), 2) if affected.any() else 0.0,
        "last_vs_mean_abs_w_p99_in_affected": round(float(gap[affected].quantile(0.99)), 2) if affected.any() else 0.0,
        "last_vs_mean_abs_w_max": round(float(gap.max()), 1),
        "first_vs_last_abs_w_max": round(float(gap_first.max()), 1),
        "energy_ratio_last_over_mean": round(float(means["last"].sum() / max(means["mean"].sum(), 1e-9)), 5),
        "energy_ratio_first_over_mean": round(float(means["first"].sum() / max(means["mean"].sum(), 1e-9)), 5),
    }


def conflict_time_split(df: pd.DataFrame, conflicts: pd.DataFrame) -> dict:
    """How often rows that conflict under one Unix label carry different Time values."""
    timestamps = conflicts["Unix"].unique()
    rows = df.loc[df["Unix"].isin(timestamps), ["Unix", "Time"]]
    differs = rows.groupby("Unix")["Time"].nunique() > 1
    return {"timestamps_with_a_conflict": len(timestamps),
            "with_differing_time_pct": round(float(differs.mean() * 100), 1) if len(timestamps) else 0.0}


def independent_readings(df: pd.DataFrame) -> dict:
    """Polls per minute, counting rows under 5 seconds apart as one burst."""
    unix = np.sort(df["Unix"].to_numpy())
    bursts = pd.Series(np.r_[True, np.diff(unix) >= 5]).groupby(unix // 60).sum()
    unique = pd.Series(np.unique(unix))
    return {
        "rows_per_minute_median": float(pd.Series(unix).groupby(unix // 60).size().median()),
        "unique_timestamps_per_minute_median": float(unique.groupby(unique // 60).size().median()),
        "polls_per_minute_median": float(bursts.median()),
    }


def nan_behaviour(df: pd.DataFrame, sensor: str) -> dict:
    """Test whether NaN means "no change", as the readme suggests, or a missed report."""
    values = df[sensor].to_numpy()
    unix = df["Unix"].to_numpy()
    reported = np.flatnonzero(~np.isnan(values))
    jump = np.flatnonzero(np.diff(reported) > 1)
    length = unix[reported[jump + 1]] - unix[reported[jump]]
    before, after = values[reported[jump]], values[reported[jump + 1]]
    buckets = []
    for label, lo, hi in (("up to 2 min", 0, 120), ("2 to 10 min", 120, 600), ("10 to 60 min", 600, 3600), ("over 60 min", 3600, np.inf)):
        here = (length > lo) & (length <= hi)
        buckets.append({"length": label, "silences": int(here.sum()), "hours": round(float(length[here].sum() / 3600), 1),
                        "same_value_both_sides_pct": round(float(np.mean(before[here] == after[here]) * 100), 1) if here.any() else None,
                        "machine_active_either_side_pct": round(float(np.mean((before[here] > 20) | (after[here] > 20)) * 100), 1) if here.any() else None})
    was_on = (~np.isnan(values[:-1])) & (values[:-1] > 20)
    was_off = (~np.isnan(values[:-1])) & (values[:-1] <= 20)
    return {
        "sensor": sensor,
        "silence_seconds_p50": float(np.percentile(length, 50)),
        "silence_seconds_p99": float(np.percentile(length, 99)),
        "nan_rate_after_active_reading_pct": round(float(np.isnan(values[1:][was_on]).mean() * 100), 2),
        "nan_rate_after_off_reading_pct": round(float(np.isnan(values[1:][was_off]).mean() * 100), 2),
        "buckets": buckets,
    }


# --------------------------------------------------------------------------- part boundary
def minute_table(df: pd.DataFrame, sensors: list[str] = SENSORS) -> pd.DataFrame:
    """Mean of available readings per raw clock minute. Used only for comparisons."""
    frame = df[["Unix", *sensors]].copy()
    frame["minute"] = frame.pop("Unix") // 60
    return frame.groupby("minute", sort=True).mean()


def best_lag(x: pd.Series, y: pd.Series, grid: np.ndarray, lags) -> tuple[int | None, float | None, int]:
    """Lag in minutes of y relative to x with the highest correlation over grid."""
    xv = x.reindex(grid).to_numpy()
    best, best_r, used = None, None, 0
    for lag in lags:
        yv = y.reindex(grid + lag).to_numpy()
        ok = ~np.isnan(xv) & ~np.isnan(yv)
        if ok.sum() < 300 or np.std(xv[ok]) == 0 or np.std(yv[ok]) == 0:
            continue
        r = float(np.corrcoef(xv[ok], yv[ok])[0, 1])
        if best_r is None or r > best_r:
            best, best_r, used = lag, r, int(ok.sum())
    return best, best_r, used


def daily_part_lag(m1: pd.DataFrame, m2: pd.DataFrame, lags=range(-120, 121)) -> pd.DataFrame:
    """For each day both parts cover, find the clock offset of Part 2 relative to Part 1."""
    start = int(max(m1.index.min(), m2.index.min()))
    end = int(min(m1.index.max(), m2.index.max()))
    rows = []
    for day_start in range(start - start % 1440, end + 1, 1440):
        grid = np.arange(day_start, day_start + 1440)
        lag, r, used = best_lag(m1["Aggregate"], m2["Aggregate"], grid, lags)
        rows.append({"day": str(pd.Timestamp(day_start * 60, unit="s").date()), "part2_lag_min": lag,
                     "r": None if r is None else round(r, 4), "minutes_compared": used})
    return pd.DataFrame(rows)


def channel_agreement(m1: pd.DataFrame, m2: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """Compare each sensor between the parts, and test that channels keep their identity."""
    grid = np.arange(lo // 60, hi // 60)
    a, b = m1.reindex(grid), m2.reindex(grid)
    rows = []
    for sensor in SENSORS:
        x, y = a[sensor].to_numpy(), b[sensor].to_numpy()
        ok = ~np.isnan(x) & ~np.isnan(y)
        same = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 300 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0 else np.nan
        scores = {}
        for other in SENSORS:
            z = a[other].to_numpy()
            both = ~np.isnan(z) & ~np.isnan(y)
            if both.sum() > 300 and np.std(z[both]) > 0 and np.std(y[both]) > 0:
                scores[other] = float(np.corrcoef(z[both], y[both])[0, 1])
        rows.append({
            "sensor": sensor,
            "minutes_part1": int((~np.isnan(x)).sum()),
            "minutes_part2": int((~np.isnan(y)).sum()),
            "minutes_both": int(ok.sum()),
            "r_same_channel": round(same, 3) if same == same else None,
            "best_part1_channel_for_part2": max(scores, key=scores.get) if scores else None,
            "mean_part1_w": round(float(np.nanmean(x)), 1) if (~np.isnan(x)).any() else None,
            "mean_part2_w": round(float(np.nanmean(y)), 1) if (~np.isnan(y)).any() else None,
            "mean_part1_when_both_w": round(float(x[ok].mean()), 1) if ok.any() else None,
            "mean_part2_when_both_w": round(float(y[ok].mean()), 1) if ok.any() else None,
            "mean_abs_diff_w": round(float(np.abs(x[ok] - y[ok]).mean()), 1) if ok.any() else None,
        })
    return pd.DataFrame(rows)


def snap_row_rates(p1: pd.DataFrame, p2: pd.DataFrame, lo: str = "2014-08-07 11:00", hi: str = "2014-08-07 19:00") -> dict:
    """Rows per 20 minutes around the 7 August snap-back, to see which file behaves oddly."""
    rates = {}
    for name, df in (("part1", p1), ("part2", p2)):
        unix = df["Unix"].to_numpy()
        window = unix[(unix >= stamp(lo)) & (unix < stamp(hi))]
        rates[name] = pd.Series(window).groupby(window // 1200).size().tolist()
    return {"from": lo, "to": hi, **rates}


def within_part_alignment(m1: pd.DataFrame, m2: pd.DataFrame, sensor: str = "Appliance5") -> list[dict]:
    """Do sensors stay aligned inside each part while the parts drift apart?"""
    rows = []
    for lo, hi in (("2014-06-28", "2014-07-10"), ("2014-07-10", "2014-07-24"), ("2014-07-24", "2014-08-08"),
                   ("2014-08-08", "2014-08-28"), ("2014-09-04", "2014-10-01")):
        grid = np.arange(stamp(lo) // 60, stamp(hi) // 60)
        lag1, r1, _ = best_lag(m1[sensor], m1["Aggregate"], grid, range(-15, 16))
        lag2, r2, _ = best_lag(m2[sensor], m2["Aggregate"], grid, range(-15, 16))
        rows.append({"from": lo, "to": hi, "part1_lag_min": lag1, "part1_r": None if r1 is None else round(r1, 3),
                     "part2_lag_min": lag2, "part2_r": None if r2 is None else round(r2, 3)})
    return rows


def part_availability(m1: pd.DataFrame, m2: pd.DataFrame, lo: int, hi: int, sensor: str = "Aggregate") -> dict:
    """Show what using Part 2 from the boundary gives up relative to Part 1."""
    grid = np.arange(lo // 60, hi // 60)
    x, y = m1[sensor].reindex(grid).notna().to_numpy(), m2[sensor].reindex(grid).notna().to_numpy()
    only1 = x & ~y
    edges = np.flatnonzero(np.diff(np.r_[0, only1.astype(int), 0]))
    runs = [(str(pd.Timestamp(int(grid[a]) * 60, unit="s")), int(b - a)) for a, b in zip(edges[::2], edges[1::2])]
    runs.sort(key=lambda item: -item[1])
    return {
        "sensor": sensor,
        "window_minutes": len(grid),
        "both": int((x & y).sum()),
        "part1_only": int(only1.sum()),
        "part2_only": int((~x & y).sum()),
        "neither": int((~x & ~y).sum()),
        "longest_part1_only_runs": runs[:6],
    }


def switch_continuity(p1: pd.DataFrame, p2: pd.DataFrame, m1: pd.DataFrame, m2: pd.DataFrame) -> dict:
    """Look for a level step where the cleaned series would change source."""
    first_part2 = int(p2["Unix"].min())
    switch_minute = first_part2 // 60
    before = m1["Aggregate"].reindex(np.arange(switch_minute - 60, switch_minute)).dropna()
    after = m2["Aggregate"].reindex(np.arange(switch_minute, switch_minute + 60)).dropna()
    before_part1 = p1.loc[p1["Unix"] < first_part2, "Unix"]
    daily = pd.concat([m1["Aggregate"].reindex(np.arange(switch_minute - 1440, switch_minute)),
                       m2["Aggregate"].reindex(np.arange(switch_minute, switch_minute + 1440))])
    typical_step = daily.diff().abs().dropna()
    last_part1 = m1["Aggregate"].reindex([switch_minute - 1]).iloc[0]
    first_part2_value = m2["Aggregate"].reindex([switch_minute]).iloc[0]
    return {
        "first_part2_timestamp": str(pd.Timestamp(first_part2, unit="s")),
        "last_part1_timestamp_before_it": str(pd.Timestamp(int(before_part1.max()), unit="s")),
        "seconds_between": int(first_part2 - before_part1.max()),
        "last_part1_minute_w": round(float(last_part1), 1),
        "first_part2_minute_w": round(float(first_part2_value), 1),
        "step_at_switch_w": round(float(abs(first_part2_value - last_part1)), 1),
        "typical_minute_step_w_p50": round(float(typical_step.median()), 1),
        "typical_minute_step_w_p95": round(float(typical_step.quantile(0.95)), 1),
        "mean_hour_before_w": round(float(before.mean()), 1),
        "mean_hour_after_w": round(float(after.mean()), 1),
    }


# --------------------------------------------------------------------------- clock changes
def transition_rows(df: pd.DataFrame, transitions: dict[str, str] = DST_TRANSITIONS) -> list[dict]:
    """Rows per clock hour around each UK clock change, as recorded in this file."""
    rows = []
    for day, direction in transitions.items():
        start = stamp(day)
        counts = [int(((df["Unix"] >= start + h * 3600) & (df["Unix"] < start + (h + 1) * 3600)).sum()) for h in range(4)]
        rows.append({"day": day, "direction": direction, "rows_in_clock_hours_00_to_03": counts})
    return rows


def backward_steps(df: pd.DataFrame) -> list[dict]:
    """List every place where the clock moves backwards in file order."""
    u = df["Unix"].to_numpy()
    steps = np.flatnonzero(np.diff(u) < 0)
    return [{"file_row": int(i), "from": str(pd.Timestamp(int(u[i]), unit="s")),
             "to": str(pd.Timestamp(int(u[i + 1]), unit="s")), "seconds": int(u[i + 1] - u[i])} for i in steps]


# --------------------------------------------------------------------------- stage entry point
def run_checks(p1: pd.DataFrame, p2: pd.DataFrame) -> dict:
    """Run every pre-cleaning check and return a JSON-ready result."""
    conflicts = {1: find_conflicts(p1), 2: find_conflicts(p2)}
    after_spikes = {1: find_conflicts(p1, limit=APPLIANCE_LIMIT_W), 2: find_conflicts(p2, limit=APPLIANCE_LIMIT_W)}
    m1, m2 = minute_table(p1), minute_table(p2)
    lo, hi = int(p2["Unix"].min()), int(p1["Unix"].max())
    week = 7 * 86400
    result = {
        "duplicates": {
            f"part{n}": {
                "overview": duplicate_overview(df),
                "conflicts": conflict_summary(conflicts[n]).to_dict("records"),
                "conflicts_after_spike_removal": conflict_summary(after_spikes[n]).to_dict("records"),
                "neighbour_evidence": [neighbour_evidence(df, s) for s in SENSORS],
                "rule_impact": [rule_impact(df, s) for s in SENSORS],
                "conflict_time_split": conflict_time_split(df, conflicts[n]),
                "independent_readings": independent_readings(df),
            } for n, df in ((1, p1), (2, p2))
        },
        "part2_nan_behaviour": nan_behaviour(p2, "Appliance5"),
        "boundary": {
            "first_part2_timestamp": str(pd.Timestamp(lo, unit="s")),
            "last_part1_timestamp": str(pd.Timestamp(hi, unit="s")),
            "overlap_days": round((hi - lo) / 86400, 1),
            "switch_continuity": switch_continuity(p1, p2, m1, m2),
            "daily_lag": daily_part_lag(m1, m2).to_dict("records"),
            "within_part_alignment": within_part_alignment(m1, m2),
            "snap_row_rates": snap_row_rates(p1, p2),
            "channel_agreement_first_week": channel_agreement(m1, m2, lo, lo + week).to_dict("records"),
            "channel_agreement_overlap": channel_agreement(m1, m2, lo, hi).to_dict("records"),
            "availability_aggregate": part_availability(m1, m2, lo, hi),
            "availability_washing_machine": part_availability(m1, m2, lo, hi, "Appliance5"),
            "availability_heater": part_availability(m1, m2, lo, hi, "Appliance9"),
        },
        "clock_changes": {
            "part1_rows": transition_rows(p1),
            "part2_rows": transition_rows(p2),
            "part1_backward_steps": backward_steps(p1),
            "part2_backward_steps": backward_steps(p2),
        },
    }
    return result
