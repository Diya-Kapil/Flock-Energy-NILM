"""Synthetic REFIT-style data, so tests run in seconds with no dataset download."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cleaning.raw_checks import stamp
from src.config import RAW_COLUMNS, SENSORS


def make_part(rows: list[dict], part: int) -> pd.DataFrame:
    """Build a raw part from {clock: "...", any sensor: value}. Unset sensors read 0 W.

    Set a sensor to None for a missing reading, or pass "unix" for a repeated timestamp.
    """
    records = []
    for row in rows:
        unix = row.get("unix", stamp(row["clock"]) if "clock" in row else None)
        record = {"Time": pd.Timestamp(unix, unit="s"), "Unix": unix}
        for sensor in SENSORS:
            value = row.get(sensor, 0.0)
            record[sensor] = np.nan if value is None else float(value)
        records.append(record)
    frame = pd.DataFrame(records, columns=RAW_COLUMNS)
    frame["Time"] = frame["Time"].astype("datetime64[s]")
    frame["Unix"] = frame["Unix"].astype("int64")
    frame["part"] = np.int8(part)
    frame["file_row"] = np.arange(len(frame), dtype="int64")
    return frame


def steady_rows(start: str, minutes: int, every_s: int = 10, **values) -> list[dict]:
    """Regular readings for a span of minutes, with a slightly varying aggregate."""
    first = stamp(start)
    return [{"unix": first + i * every_s, "Aggregate": 200 + (i % 7), **values}
            for i in range(minutes * 60 // every_s)]


@pytest.fixture
def small_parts() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A short winter Part 1 followed by an overlapping Part 2, away from any clock change."""
    part1 = make_part(steady_rows("2014-01-10 12:00", 30, Appliance5=100), 1)
    part2 = make_part(steady_rows("2014-01-10 12:20", 30, Appliance5=100), 2)
    return part1, part2


def csv_text(frame: pd.DataFrame) -> str:
    """The frame as raw CSV text, written the way REFIT writes missing readings."""
    out = frame[RAW_COLUMNS].copy()
    out["Time"] = out["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out.to_csv(index=False, na_rep="NaN")
