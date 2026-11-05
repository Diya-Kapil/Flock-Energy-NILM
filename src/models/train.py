"""Leave-one-household-out training for the Random Forest disaggregation model.

Houses are never mixed across the split, so no time window appears in both training and
evaluation. All choices (sampling mix and on/off threshold) use training houses only.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import SEED
from src.models.dataset import HouseData, training_matrix
from src.models.metrics import ON_W, best_threshold, on_off

OFF_PER_ACTIVE_OPTIONS = (None, 4.0, 1.0)
PREDICTION_DECIMALS = 6


@dataclass(frozen=True)
class ForestSettings:
    n_estimators: int = 100
    min_samples_leaf: int = 10
    max_features: float = 0.33
    rows_per_house: int = 60000


FINAL = ForestSettings()
INNER = ForestSettings(n_estimators=20, rows_per_house=20000)


def fit_forest(X: pd.DataFrame, y: np.ndarray, settings: ForestSettings, seed: int) -> RandomForestRegressor:
    forest = RandomForestRegressor(n_estimators=settings.n_estimators, min_samples_leaf=settings.min_samples_leaf,
                                   max_features=settings.max_features, random_state=seed, n_jobs=-1)
    return forest.fit(X, y)


def predict(forest, X: pd.DataFrame) -> np.ndarray:
    """Predictions rounded to remove parallel floating-point noise, so reruns match exactly."""
    return np.round(forest.predict(X), PREDICTION_DECIMALS)


def select_on_training_houses(houses: list[HouseData], seed: int, options=OFF_PER_ACTIVE_OPTIONS,
                              settings: ForestSettings = INNER) -> tuple[pd.DataFrame, float | None, float]:
    """Choose the sampling mix and on/off threshold by inner leave-one-house-out.

    The outer held-out house is not passed in, so it cannot influence either choice.
    """
    rows = []
    for option_index, off_per_active in enumerate(options):
        actual, predicted = [], []
        for inner_index, inner in enumerate(houses):
            others = [h for h in houses if h.house != inner.house]
            X, y = training_matrix(others, settings.rows_per_house, off_per_active, seed + 100 * option_index + inner_index)
            forest = fit_forest(X, y, settings, seed + option_index)
            actual.append(inner.y)
            predicted.append(predict(forest, inner.X))
        y_all, p_all = np.concatenate(actual), np.concatenate(predicted)
        threshold, f1 = best_threshold(y_all, p_all)
        result = on_off(y_all > ON_W, p_all > threshold)
        rows.append({"off_per_active": "natural" if off_per_active is None else off_per_active, "threshold_w": threshold,
                     "inner_f1": f1, "inner_precision": result["precision"], "inner_recall": result["recall"],
                     "inner_mae_w": float(np.abs(p_all - y_all).mean()),
                     "inner_energy_ratio": float(p_all.sum() / y_all.sum())})
    table = pd.DataFrame(rows)
    best = table.sort_values(["inner_f1", "inner_mae_w"], ascending=[False, True]).iloc[0]
    chosen = None if best["off_per_active"] == "natural" else float(best["off_per_active"])
    return table, chosen, float(best["threshold_w"])


def impurity_importance(forest: RandomForestRegressor, columns) -> dict[str, float]:
    return {str(name): round(float(value), 6) for name, value in zip(columns, forest.feature_importances_)}


def run_fold(test: HouseData, train: list[HouseData], settings: ForestSettings, off_per_active: float | None, seed: int):
    """Fit on training houses and predict every usable minute of the held-out house."""
    X, y = training_matrix(train, settings.rows_per_house, off_per_active, seed)
    started = time.time()
    forest = fit_forest(X, y, settings, seed)
    fit_seconds = time.time() - started
    assert test.house not in {h.house for h in train}
    return forest, predict(forest, test.X), {"training_rows": len(y), "training_active_share": float((y > ON_W).mean()),
                                             "fit_seconds": round(fit_seconds, 1)}


def save_model(forest, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(forest, path, compress=3)


def settings_dict(settings: ForestSettings = FINAL) -> dict:
    return {**asdict(settings), "seed": SEED}
