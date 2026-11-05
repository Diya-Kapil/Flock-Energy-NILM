"""Assemble model inputs per household, and sample training rows from training houses only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.features import make_features
from src.models.metrics import ON_W

MIN_BIN_COVERAGE = 0.8


@dataclass
class HouseData:
    house: int
    positions: np.ndarray  # Grid positions of usable rows, to map results back to the table.
    X: pd.DataFrame
    y: np.ndarray


def build_house(table: pd.DataFrame, house: int, exclude_frozen: bool = True, step_minutes: int = 1,
                retrospective: bool = False) -> HouseData:
    """Features and target for the usable rows of one house.

    Excluded minutes become NaN before features are made, so frozen values can't leak
    into the rolling windows of nearby minutes.
    """
    usable = (table["valid"] if exclude_frozen else ~table["missing"]).to_numpy()
    features = make_features(table["aggregate_w"].where(usable).to_numpy(dtype="float64"), step_minutes, retrospective)
    positions = np.flatnonzero(usable)
    return HouseData(house=house, positions=positions, X=features.iloc[positions].reset_index(drop=True),
                     y=table["washing_machine_w"].to_numpy(dtype="float32")[positions])


def sample_rows(y: np.ndarray, rows: int, off_per_active: float | None, rng: np.random.Generator, on_w: float = ON_W) -> np.ndarray:
    """Choose training rows from one house.

    None keeps the natural mix of on and off minutes. Otherwise active minutes are kept up
    to the share implied by the ratio, which gives trees more operating examples to learn from.
    """
    n = len(y)
    if off_per_active is None:
        return np.sort(rng.choice(n, size=min(rows, n), replace=False))
    active, off = np.flatnonzero(y > on_w), np.flatnonzero(y <= on_w)
    n_active = min(len(active), int(rows / (1 + off_per_active)))
    n_off = min(len(off), int(round(n_active * off_per_active)))
    chosen_active = rng.choice(active, size=n_active, replace=False)
    chosen_off = rng.choice(off, size=n_off, replace=False)
    return np.sort(np.r_[chosen_active, chosen_off])


def training_matrix(houses: list[HouseData], rows_per_house: int, off_per_active: float | None, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Stack sampled rows from the given houses. Each house contributes equally."""
    rng = np.random.default_rng(seed)
    frames, targets = [], []
    for data in houses:
        chosen = sample_rows(data.y, rows_per_house, off_per_active, rng)
        frames.append(data.X.iloc[chosen])
        targets.append(data.y[chosen])
    return pd.concat(frames, ignore_index=True), np.concatenate(targets)


def downsample(table: pd.DataFrame, step_minutes: int, exclude_frozen: bool = True) -> pd.DataFrame:
    """Average to a coarser interval, as a smart meter would report it.

    An interval is usable only if at least 80% of its minutes are usable, and it keeps
    the mean of those minutes. Bins are aligned to the official clock.
    """
    usable = table["valid"] if exclude_frozen else ~table["missing"]
    seconds = (table["official_minute"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    bins = (seconds.to_numpy() // (step_minutes * 60))
    frame = pd.DataFrame({"bin": bins, "aggregate_w": table["aggregate_w"].where(usable).astype("float64"),
                          "washing_machine_w": table["washing_machine_w"].where(usable).astype("float64"),
                          "issues_any": table["issues_any"].to_numpy(), "usable": usable.to_numpy()})
    grouped = frame.groupby("bin", sort=True)
    means = grouped[["aggregate_w", "washing_machine_w"]].mean()
    coverage = grouped["usable"].mean()
    keep = coverage >= MIN_BIN_COVERAGE
    grid = np.arange(means.index.min(), means.index.max() + 1)
    means = means.where(keep).reindex(grid)
    return pd.DataFrame({
        "official_minute": pd.to_datetime(grid * step_minutes * 60, unit="s"),
        "aggregate_w": means["aggregate_w"].to_numpy(dtype="float32"),
        "washing_machine_w": means["washing_machine_w"].to_numpy(dtype="float32"),
        "issues_any": grouped["issues_any"].any().reindex(grid, fill_value=False).to_numpy(),
        "missing": means["aggregate_w"].isna().to_numpy(),
        "valid": means["aggregate_w"].notna().to_numpy(),
    })
