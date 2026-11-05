"""Turn official cleaned CSVs into 1-minute tables for analysis and modelling.

Original timestamps are kept. The UK local time is a derived column for hourly analysis
only. Values are never changed: frozen and Issues minutes are flagged, and each analysis
decides what to exclude.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.cleaning.clock import to_uk_local
from src.cleaning.raw_house1 import frozen_mask
from src.config import AGGREGATE_REVIEW_W, APPLIANCES, FROZEN_MIN_APPLIANCE_W, HOUSEHOLDS, SENSORS


def official_to_minutes(official: pd.DataFrame, house: int, daily: pd.DataFrame) -> pd.DataFrame:
    """Mean per official clock minute on a complete grid, plus quality flags.

    Aggregate and appliances come from the same rows, so they stay aligned. Minutes
    without rows stay missing, and nothing is filled.
    """
    machine = f"Appliance{HOUSEHOLDS[house]}"
    minute = official["Unix"].to_numpy() // 60
    means = official[SENSORS].astype("float64").groupby(minute).mean()
    grid = np.arange(minute.min(), minute.max() + 1)
    means = means.reindex(grid)
    rows_per_minute = pd.Series(1, index=minute).groupby(level=0).sum().reindex(grid, fill_value=0)
    issues = pd.Series(official["Issues"].to_numpy(), index=minute).groupby(level=0).sum().reindex(grid, fill_value=0)
    over_review = pd.Series((official["Aggregate"] > AGGREGATE_REVIEW_W).to_numpy().astype(int), index=minute)
    over_review = over_review.groupby(level=0).sum().reindex(grid, fill_value=0)

    aggregate = means["Aggregate"].to_numpy()
    washing_machine = means[machine].to_numpy()
    aggregate_frozen = frozen_mask(grid, aggregate, None)
    machine_frozen = frozen_mask(grid, washing_machine, FROZEN_MIN_APPLIANCE_W)
    missing = np.isnan(aggregate) | np.isnan(washing_machine)
    frozen = aggregate_frozen | machine_frozen

    local = to_uk_local(grid * 60, daily)
    frame = pd.DataFrame({
        "official_minute": pd.to_datetime(grid * 60, unit="s"),
        "uk_local_minute": pd.to_datetime(local["uk_local_seconds"].astype("float64"), unit="s"),
        "clock_status": local["clock_status"].fillna("no_offset").astype("string").to_numpy(),
        "aggregate_w": aggregate.astype("float32"),
        "washing_machine_w": washing_machine.astype("float32"),
        "n_readings": rows_per_minute.to_numpy().astype("int16"),
        "issues_rows": issues.to_numpy().astype("int16"),
        "issues_any": issues.to_numpy() > 0,
        "aggregate_over_23kw": over_review.to_numpy().astype("int16"),
        "missing": missing,
        "aggregate_frozen": aggregate_frozen,
        "washing_machine_frozen": machine_frozen,
        "frozen": frozen,
        "valid": ~missing & ~frozen,
    })
    others = {f"other_{name.lower()}_w": means[name].to_numpy().astype("float32") for name in APPLIANCES if name != machine}
    return pd.concat([frame, pd.DataFrame(others)], axis=1)
