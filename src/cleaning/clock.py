"""Measure the official cleaned clock against the raw UK local clock, per house and day.

The official files shift some periods by an hour and not others. This module records the
offset for each day and derives a UK local time for hourly analysis. It never changes the
official files or their timestamps, and aggregate and appliance columns share one row, so
they stay aligned whatever the offset.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from src.cleaning.raw_checks import daily_part_lag, stamp
from src.config import DST_TRANSITIONS

CLOCK_LAGS = (-60, 0, 60)
MIN_R = 0.9
MIN_MARGIN = 0.02
MIN_MINUTES = 360
HOURS_TOLERANCE_MINUTES = 2
USABLE = ("verified", "inferred")


def raw_member_name(house: int, part: int) -> str:
    """Raw file name inside the archive. Houses 13 and above have an extra underscore."""
    separator = "_" if house >= 13 else ""
    return f"RAW_House{separator}{house}_Part{part}.csv"


def clean_member_name(house: int) -> str:
    return f"CLEAN_House{house}.csv"


def read_minute_aggregate(data: bytes) -> pd.Series:
    """Mean aggregate per clock minute, read from the bytes of one REFIT CSV."""
    frame = pd.read_csv(io.BytesIO(data), engine="pyarrow", usecols=["Unix", "Aggregate"]).dropna()
    return frame["Aggregate"].astype("float64").groupby(frame["Unix"].to_numpy() // 60).mean()


def combine_raw(part1: pd.Series, part2: pd.Series) -> pd.Series:
    """Prefer Part 1 where both exist, because that is what the official files mostly used."""
    combined = pd.concat([part1, part2])
    return combined[~combined.index.duplicated(keep="first")].sort_index()


def daily_offsets(clean: pd.Series, raw: pd.Series, lags=CLOCK_LAGS) -> pd.DataFrame:
    """For each cleaned-file day, find clean_minus_raw_min: cleaned label minus raw label."""
    rows = []
    for day in np.unique(clean.index.to_numpy() // 1440):
        grid = np.arange(day * 1440, (day + 1) * 1440)
        x = clean.reindex(grid).to_numpy()
        scores: dict[int, tuple[float, int]] = {}
        for lag in lags:
            y = raw.reindex(grid - lag).to_numpy()
            ok = ~np.isnan(x) & ~np.isnan(y)
            if ok.sum() < MIN_MINUTES or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
                continue
            scores[lag] = (float(np.corrcoef(x[ok], y[ok])[0, 1]), int(ok.sum()))
        record = {"day": str(pd.Timestamp(int(day) * 86400, unit="s").date()), "clean_minutes": int((~np.isnan(x)).sum()),
                  "clean_minus_raw_min": np.nan, "r": np.nan, "margin": np.nan, "status": "uncertain"}
        if scores:
            ranked = sorted(scores.items(), key=lambda kv: -kv[1][0])
            best, (best_r, _) = ranked[0]
            margin = best_r - ranked[1][1][0] if len(ranked) > 1 else 1.0
            record.update({"clean_minus_raw_min": best, "r": round(best_r, 4), "margin": round(margin, 4)})
            if best_r >= MIN_R and margin >= MIN_MARGIN:
                record["status"] = "verified"
        rows.append(record)
    return pd.DataFrame(rows)


def classify_days(daily: pd.DataFrame) -> pd.DataFrame:
    """Fill weak days only between agreeing neighbours, and mark days around each change.

    A day next to a change in offset can hold rows from both offsets, so both such days get
    status "transition" and keep their measured offset for validation only.
    """
    table = daily.copy()
    offset = table["clean_minus_raw_min"].where(table["status"] == "verified")
    before, after = offset.ffill(), offset.bfill()
    agree = (table["status"] != "verified") & (before == after)
    table.loc[agree, "clean_minus_raw_min"] = before[agree]
    table.loc[agree, "status"] = "inferred"
    table.loc[(table["status"] == "uncertain"), "clean_minus_raw_min"] = np.nan

    known = table["clean_minus_raw_min"].where(table["status"].isin(USABLE))
    filled_previous, filled_next = known.ffill(), known.bfill()
    next_differs = filled_previous.notna() & filled_next.shift(-1).notna() & (filled_previous != filled_next.shift(-1))
    change_after = np.flatnonzero(next_differs.to_numpy())
    usable = table["status"].isin(USABLE).to_numpy()
    for index in change_after:
        table.loc[index, "status"] = "transition" if usable[index] else table.loc[index, "status"]
        following = index + 1
        if following < len(table) and usable[following]:
            table.loc[following, "status"] = "transition"
    return table


def boundaries(daily: pd.DataFrame, house: int) -> pd.DataFrame:
    """Compress the daily table into periods, the form documented in DATA.md."""
    table = daily.copy()
    table["group"] = np.where(table["status"].isin(USABLE), "usable", table["status"])
    # A sentinel, because NaN never equals itself and would split each uncertain day into a run.
    key = table["group"] + "|" + table["clean_minus_raw_min"].fillna(-999).astype("int64").astype(str)
    runs = (key != key.shift()).cumsum()
    rows = []
    for _, block in table.groupby(runs, sort=True):
        rows.append({
            "house": house,
            "start_day": block["day"].iloc[0],
            "end_day": block["day"].iloc[-1],
            "days": len(block),
            "clean_minus_raw_min": block["clean_minus_raw_min"].iloc[0],
            "to_get_uk_local_add_min": None if pd.isna(block["clean_minus_raw_min"].iloc[0]) else 0 - block["clean_minus_raw_min"].iloc[0],
            "status": block["group"].iloc[0],
            "days_verified": int((block["status"] == "verified").sum()),
            "days_inferred": int((block["status"] == "inferred").sum()),
            "median_r": round(float(block["r"].median()), 4) if block["r"].notna().any() else None,
        })
    return pd.DataFrame(rows)


def to_uk_local(clean_unix: np.ndarray, daily: pd.DataFrame, include_transition: bool = False) -> pd.DataFrame:
    """Derive UK local clock seconds for hourly analysis. The input array is left untouched.

    Rows on uncertain days, and by default transition days, get a missing local time so
    they cannot enter an hourly analysis unnoticed.
    """
    allowed = ("verified", "inferred", "transition") if include_transition else USABLE
    days = (pd.to_datetime(daily["day"]) - pd.Timestamp("1970-01-01")) // pd.Timedelta(days=1)
    lookup = pd.Series(daily["clean_minus_raw_min"].to_numpy(), index=days.to_numpy())
    status = pd.Series(daily["status"].to_numpy(), index=days.to_numpy())
    day = clean_unix // 86400
    row_status = status.reindex(day).to_numpy()
    offset = lookup.reindex(day).to_numpy()
    usable = np.isin(row_status, allowed) & ~pd.isna(offset)
    local = pd.array(np.where(usable, clean_unix - np.nan_to_num(offset).astype("int64") * 60, 0), dtype="Int64")
    local[~usable] = pd.NA
    return pd.DataFrame({"uk_local_seconds": local, "clock_status": row_status})


def block_offsets(clean: pd.Series, part1: pd.Series, part2: pd.Series, lo: str, hi: str, block_minutes: int = 120) -> list[dict]:
    """Finer view of a change: the best source and offset for each short block of a day."""
    grid = np.arange(stamp(lo) // 60, stamp(hi) // 60)
    rows = []
    for start in range(0, len(grid), block_minutes):
        block = grid[start:start + block_minutes]
        x = clean.reindex(block).to_numpy()
        best = None
        for name, raw in (("part1", part1), ("part2", part2)):
            for lag in CLOCK_LAGS:
                y = raw.reindex(block - lag).to_numpy()
                ok = ~np.isnan(x) & ~np.isnan(y)
                if ok.sum() < 20 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
                    continue
                r = float(np.corrcoef(x[ok], y[ok])[0, 1])
                if best is None or r > best["r"]:
                    best = {"source": name, "clean_minus_raw_min": lag, "r": round(r, 3)}
        rows.append({"official_clock_block": str(pd.Timestamp(int(block[0]) * 60, unit="s")), "official_minutes": int((~np.isnan(x)).sum()),
                     **(best or {"source": None, "clean_minus_raw_min": None, "r": None})})
    return rows


def part_drift_by_house(part1: pd.Series, part2: pd.Series) -> dict:
    """Largest Part 2 lag behind Part 1 in summer 2014, to see whether drift is shared."""
    lags = daily_part_lag(part1.to_frame("Aggregate"), part2.to_frame("Aggregate"), range(-15, 16))
    good = lags[(lags["r"] >= 0.8) & (lags["part2_lag_min"].notna())]
    return {
        "days_compared": len(good),
        "days_with_lag": int((good["part2_lag_min"] != 0).sum()),
        "max_lag_min": int(good["part2_lag_min"].max()) if len(good) else None,
        "lag_on_2014_08_05": next(iter(good.loc[good["day"] == "2014-08-05", "part2_lag_min"].astype(int)), None),
        "lag_on_2014_09_10": next(iter(good.loc[good["day"] == "2014-09-10", "part2_lag_min"].astype(int)), None),
    }


def hour_minutes(series_minutes: np.ndarray, day: str) -> list[int]:
    """Distinct clock minutes with data in clock hours 00 to 03 of a date."""
    start = stamp(day) // 60
    return [int(np.unique(series_minutes[(series_minutes >= start + h * 60) & (series_minutes < start + (h + 1) * 60)]).size)
            for h in range(4)]


def validate_transitions(clean_unix: np.ndarray, raw_minutes: np.ndarray, daily: pd.DataFrame) -> list[dict]:
    """After correction, each clock change should look the same as it does in raw data.

    Hour 00 of a spring change is reported but not judged: the official files move that hour
    into the previous day, where it interleaves with other rows.
    """
    local = to_uk_local(clean_unix, daily, include_transition=True)["uk_local_seconds"]
    known = local.notna().to_numpy()
    corrected = local[known].to_numpy(dtype="int64") // 60
    results = []
    for day, direction in DST_TRANSITIONS.items():
        raw_hours = hour_minutes(raw_minutes, day)
        fixed_hours = hour_minutes(corrected, day)
        uncorrected = hour_minutes(clean_unix // 60, day)
        judged = range(1, 4) if direction == "forward" else range(4)
        if sum(raw_hours) == 0:
            verdict = "no_data"
        elif all(abs(raw_hours[h] - fixed_hours[h]) <= HOURS_TOLERANCE_MINUTES for h in judged):
            verdict = "pass"
        else:
            verdict = "review"
        results.append({"day": day, "direction": direction, "raw_minutes_by_hour": raw_hours,
                        "official_uncorrected_minutes_by_hour": uncorrected,
                        "corrected_minutes_by_hour": fixed_hours, "verdict": verdict})
    return results


def routine_shift(minute_values: pd.Series) -> int | None:
    """Minutes by which the spring 2015 daily pattern sits later than spring 2014's."""
    profiles = []
    for lo, hi in (("2014-04-05", "2014-05-31"), ("2015-04-05", "2015-05-31")):
        window = minute_values[(minute_values.index >= stamp(lo) // 60) & (minute_values.index < stamp(hi) // 60)]
        if len(window) < 3000:
            return None
        profiles.append(window.groupby((window.index % 1440) // 5).mean().reindex(range(288))
                        .interpolate(limit_direction="both").to_numpy())
    scores = [np.corrcoef(profiles[0], np.roll(profiles[1], -k))[0, 1] for k in range(-24, 25)]
    return (int(np.argmax(scores)) - 24) * 5


def corrected_minute_series(clean: pd.Series, daily: pd.DataFrame) -> pd.Series:
    """Move a minute series onto UK local labels, dropping days without a usable offset."""
    local = to_uk_local(clean.index.to_numpy() * 60, daily)["uk_local_seconds"]
    known = local.notna().to_numpy()
    moved = pd.Series(clean.to_numpy()[known], index=local[known].to_numpy(dtype="int64") // 60)
    return moved.groupby(level=0).mean().sort_index()
