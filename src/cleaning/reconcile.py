"""Compare cleaned House 1 with the official cleaned file, on the same clock.

The aim is to explain differences, not to match the official file. Some differences are
deliberate: spikes become missing here and zero there, and long silences stay missing here
while the official file holds the last value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.cleaning.clock import to_uk_local
from src.cleaning.raw_checks import stamp
from src.config import APPLIANCES, HOUSE1_LABELS, SENSORS


def official_on_raw_clock(official: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Official readings as 1-minute means on UK local labels, using usable days only."""
    local = to_uk_local(official["Unix"].to_numpy(), daily)["uk_local_seconds"]
    known = local.notna().to_numpy()
    frame = official.loc[known, SENSORS].astype("float64").copy()
    frame["minute"] = local[known].to_numpy(dtype="int64") // 60
    return frame.groupby("minute").mean()


def ours_on_raw_clock(minutes: pd.DataFrame) -> pd.DataFrame:
    """Cleaned table keyed by raw clock minute. A repeated clock hour keeps its first pass."""
    frame = minutes.copy()
    frame["minute"] = (frame["raw_clock_minute"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(minutes=1)
    return frame.drop_duplicates("minute").set_index("minute")


def compare_period(ours: pd.DataFrame, official: pd.DataFrame, lo: int, hi: int) -> list[dict]:
    """Agreement per sensor over a window of clock minutes."""
    grid = np.arange(lo // 60, hi // 60)
    mine, theirs = ours.reindex(grid), official.reindex(grid)
    out = []
    for sensor in SENSORS:
        x, y = mine[sensor].to_numpy(dtype="float64"), theirs[sensor].to_numpy(dtype="float64")
        have_x, have_y = ~np.isnan(x), ~np.isnan(y)
        both = have_x & have_y
        varied = both.sum() > 300 and np.std(x[both]) > 0 and np.std(y[both]) > 0
        out.append({
            "sensor": sensor,
            "label": HOUSE1_LABELS[sensor],
            "minutes_both": int(both.sum()),
            "minutes_only_ours": int((have_x & ~have_y).sum()),
            "minutes_only_official": int((~have_x & have_y).sum()),
            "r": round(float(np.corrcoef(x[both], y[both])[0, 1]), 4) if varied else None,
            "mean_abs_diff_w": round(float(np.abs(x[both] - y[both]).mean()), 2) if both.any() else None,
            "ours_mean_w": round(float(x[both].mean()), 2) if both.any() else None,
            "official_mean_w": round(float(y[both].mean()), 2) if both.any() else None,
        })
    return out


def monthly_alignment(ours: pd.DataFrame, official: pd.DataFrame) -> list[dict]:
    """Check the corrected official clock against cleaned raw data, month by month."""
    rows = []
    first, last = int(ours.index.min()), int(ours.index.max())
    months = pd.date_range(pd.Timestamp(first * 60, unit="s").normalize().replace(day=1),
                           pd.Timestamp(last * 60, unit="s"), freq="MS")
    for start in months:
        grid = np.arange(stamp(str(start)) // 60, stamp(str(start + pd.offsets.MonthBegin(1))) // 60)
        x = ours["Aggregate"].reindex(grid).to_numpy(dtype="float64")
        best = None
        for lag in (-60, 0, 60):
            y = official["Aggregate"].reindex(grid + lag).to_numpy(dtype="float64")
            ok = ~np.isnan(x) & ~np.isnan(y)
            if ok.sum() < 1000 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
                continue
            r = float(np.corrcoef(x[ok], y[ok])[0, 1])
            if best is None or r > best[1]:
                best = (lag, r, int(ok.sum()))
        rows.append({"month": start.strftime("%Y-%m"),
                     "lag_min": None if best is None else best[0],
                     "r": None if best is None else round(best[1], 4),
                     "minutes_compared": 0 if best is None else best[2]})
    return rows


def spike_treatment(ours: pd.DataFrame, official: pd.DataFrame) -> list[dict]:
    """In minutes where a spike was removed here, what does the official file show?"""
    out = []
    joined = official.reindex(ours.index)
    for sensor in APPLIANCES:
        flagged = ours[f"{sensor}_spikes"].to_numpy() > 0
        theirs = joined[sensor].to_numpy(dtype="float64")[flagged]
        mine = ours[sensor].to_numpy(dtype="float64")[flagged]
        out.append({
            "sensor": sensor,
            "minutes_with_spike_removed": int(flagged.sum()),
            "ours_missing_in_those_minutes": int(np.isnan(mine).sum()),
            "official_mean_w_in_those_minutes": round(float(np.nanmean(theirs)), 1) if np.isfinite(theirs).any() else None,
            "official_exactly_zero_minutes": int((theirs == 0).sum()),
        })
    return out


def held_values(ours: pd.DataFrame, official: pd.DataFrame, start_minute: int) -> list[dict]:
    """Where a sensor is silent here, does the official file show a value it did not measure?

    Limited to the period where both files come from Part 2, where silence can be told
    apart from a reading of zero.
    """
    out = []
    part2 = ours.index >= start_minute
    joined = official.reindex(ours.index)
    for sensor in SENSORS:
        silent = ours[f"{sensor}_missing"].to_numpy() & part2 & ~ours["dst_ambiguous"].to_numpy()
        values = joined[sensor].to_numpy(dtype="float64")[silent]
        present = values[~np.isnan(values)]
        out.append({
            "sensor": sensor,
            "silent_minutes_here": int(silent.sum()),
            "official_has_a_value": len(present),
            "official_value_is_zero": int((present == 0).sum()),
            "official_value_is_nonzero": int((present != 0).sum()),
            "official_energy_kwh_in_those_minutes": round(float(present.sum() / 60000), 2),
        })
    return out


def silence_behaviour(ours: pd.DataFrame, official: pd.DataFrame, start_minute: int, sensor: str = "Appliance5") -> list[dict]:
    """What does the official file show across a run of minutes where this sensor is silent?

    Only runs where other sensors keep reporting, in the period where both files use Part 2.
    A single repeated value across a run means the official file held the last reading.
    """
    silent = (ours[f"{sensor}_missing"].to_numpy() & (ours["outage_minutes"].to_numpy() == 0)
              & (ours.index.to_numpy() >= start_minute) & ~ours["dst_ambiguous"].to_numpy())
    edges = np.flatnonzero(np.diff(np.r_[0, silent.astype("int8"), 0]))
    joined = official[sensor].reindex(ours.index).to_numpy(dtype="float64")
    records = []
    for start, stop in zip(edges[::2].tolist(), edges[1::2].tolist()):
        values = joined[start:stop]
        values = values[~np.isnan(values)]
        if len(values):
            records.append({"length": stop - start, "single_value": bool(np.unique(values).size == 1),
                            "held_at_zero": bool((values == 0).all())})
    table = pd.DataFrame(records, columns=["length", "single_value", "held_at_zero"])
    out = []
    for label, lo, hi in (("under 10 min", 1, 10), ("10 to 59 min", 10, 60), ("60 min or more", 60, 10**9)):
        here = table[(table["length"] >= lo) & (table["length"] < hi)]
        out.append({"silence_length": label, "runs": len(here), "held_one_value": int(here["single_value"].sum()),
                    "held_at_nonzero": int((here["single_value"] & ~here["held_at_zero"]).sum())})
    return out


def reconcile(minutes: pd.DataFrame, official: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """Full comparison with the official cleaned House 1 file."""
    ours = ours_on_raw_clock(minutes)
    theirs = official_on_raw_clock(official, daily)
    part2_start = stamp(str(minutes.loc[minutes["source_part"] == 2, "raw_clock_minute"].iloc[0]))
    official_part_change = stamp("2014-10-01 02:00")
    periods = {
        "both_use_part1": (int(ours.index.min()) * 60, part2_start),
        "ours_part2_official_mostly_part1": (part2_start, official_part_change),
        "both_use_part2": (official_part_change, int(ours.index.max()) * 60 + 60),
    }
    return {
        "official_rows": len(official),
        "official_minutes_compared": len(theirs),
        "periods": {name: {"from": str(pd.Timestamp(lo, unit="s")), "to": str(pd.Timestamp(hi, unit="s")),
                           "sensors": compare_period(ours, theirs, lo, hi)} for name, (lo, hi) in periods.items()},
        "monthly_alignment": monthly_alignment(ours, theirs),
        "spike_treatment": spike_treatment(ours, theirs),
        "held_values_in_official": held_values(ours, theirs, official_part_change // 60),
        "washing_machine_silence_behaviour": silence_behaviour(ours, theirs, official_part_change // 60),
    }
