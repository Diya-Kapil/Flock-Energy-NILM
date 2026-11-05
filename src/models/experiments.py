"""Secondary experiments, each clearly separate from the primary leave-one-house-out result.

- no_frozen_exclusion: how much does excluding frozen minutes matter?
- retrospective: how much would future readings help after-the-fact analysis?
- two_stage: does an on/off classifier plus a power regressor materially help?
- 15-minute and 30-minute: how does the method degrade at smart-meter resolution?

None of them replaces the primary model. Settings match the primary model, and every choice
uses training houses only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.analysis.cycles import DEFAULT_PARAMS, detect_cycles
from src.models.dataset import HouseData, build_house, downsample, training_matrix
from src.models.evaluate import cycle_performance
from src.models.metrics import ON_W, evaluate, on_off
from src.models.train import FINAL, INNER, ForestSettings, fit_forest, predict, select_on_training_houses

PROBABILITY_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def fold_seed(base: int, fold: int) -> int:
    return base + fold


def variant_frame(test: HouseData, predicted: np.ndarray, threshold: float, table: pd.DataFrame) -> pd.DataFrame:
    positions = test.positions
    return pd.DataFrame({"grid_pos": positions, "actual_w": test.y, "predicted_w": predicted, "threshold_w": threshold,
                         "issues_any": table["issues_any"].to_numpy()[positions]})


def run_forest_variant(tables: dict[int, pd.DataFrame], folds: dict[int, dict], seed: int, reselect_threshold: bool = False,
                       reselect_all: bool = False, step_minutes: int = 1, **options) -> tuple[dict[int, pd.DataFrame], list[dict]]:
    """Run leave-one-house-out with changed inputs, otherwise as in the primary model."""
    data = {house: build_house(table, house, step_minutes=step_minutes, **options) for house, table in tables.items()}
    predictions, log = {}, []
    for fold, (house, test) in enumerate(data.items()):
        train = [d for h, d in data.items() if h != house]
        fold_info = folds[house]
        off_per_active, threshold = fold_info["chosen_off_per_active"], fold_info["threshold_w"]
        if reselect_all:
            _, off_per_active, threshold = select_on_training_houses(train, fold_seed(seed, fold))
        elif reselect_threshold:
            _, _, threshold = select_on_training_houses(train, fold_seed(seed, fold), options=(off_per_active,))
        X, y = training_matrix(train, FINAL.rows_per_house, off_per_active, fold_seed(seed, fold))
        forest = fit_forest(X, y, FINAL, fold_seed(seed, fold))
        predictions[house] = variant_frame(test, predict(forest, test.X), threshold, tables[house])
        log.append({"held_out_house": house, "off_per_active": off_per_active, "threshold_w": threshold})
    return predictions, log


def fit_two_stage(X: pd.DataFrame, y: np.ndarray, settings: ForestSettings, seed: int):
    classifier = RandomForestClassifier(n_estimators=settings.n_estimators, min_samples_leaf=settings.min_samples_leaf,
                                        max_features=settings.max_features, random_state=seed, n_jobs=-1).fit(X, y > ON_W)
    regressor = fit_forest(X[y > ON_W], y[y > ON_W], settings, seed)
    return classifier, regressor


def two_stage_predict(classifier, regressor, X: pd.DataFrame, cutoff: float) -> np.ndarray:
    """Regressor output where the classifier says on, and zero elsewhere."""
    predicted = np.zeros(len(X))
    on = classifier.predict_proba(X)[:, 1] >= cutoff
    if on.any():
        predicted[on] = predict(regressor, X[on])
    return predicted


def select_probability_cutoff(train: list[HouseData], off_per_active: float, seed: int, settings: ForestSettings = INNER) -> float:
    """Classifier cutoff with the best F1 under inner leave-one-house-out on training houses."""
    actual, probability = [], []
    for inner_index, inner in enumerate(train):
        others = [h for h in train if h.house != inner.house]
        X, y = training_matrix(others, settings.rows_per_house, off_per_active, seed + inner_index)
        classifier = RandomForestClassifier(n_estimators=settings.n_estimators, min_samples_leaf=settings.min_samples_leaf,
                                            max_features=settings.max_features, random_state=seed, n_jobs=-1).fit(X, y > ON_W)
        actual.append(inner.y > ON_W)
        probability.append(classifier.predict_proba(inner.X)[:, 1])
    active, scores = np.concatenate(actual), np.concatenate(probability)
    return max(PROBABILITY_GRID, key=lambda cutoff: on_off(active, scores >= cutoff)["f1"])


def run_two_stage(tables: dict[int, pd.DataFrame], seed: int, off_per_active: float = 4.0) -> tuple[dict[int, pd.DataFrame], list[dict]]:
    """Leave-one-house-out for the two-stage model. Predictions above zero mean predicted on."""
    data = {house: build_house(table, house) for house, table in tables.items()}
    predictions, log = {}, []
    for fold, (house, test) in enumerate(data.items()):
        train = [d for h, d in data.items() if h != house]
        cutoff = select_probability_cutoff(train, off_per_active, fold_seed(seed, fold))
        X, y = training_matrix(train, FINAL.rows_per_house, off_per_active, fold_seed(seed, fold))
        classifier, regressor = fit_two_stage(X, y, FINAL, fold_seed(seed, fold))
        predicted = np.round(two_stage_predict(classifier, regressor, test.X, cutoff), 6)
        predictions[house] = variant_frame(test, predicted, 0.0, tables[house])
        log.append({"held_out_house": house, "probability_cutoff": cutoff})
    return predictions, log


def summarise(name: str, predictions: dict[int, pd.DataFrame], step_minutes: int = 1, grids: dict[int, int] | None = None,
              label: str = "") -> list[dict]:
    """Per-house and pooled metrics for one variant, plus cycle detection at 1-minute resolution."""
    rows = []
    cycle_table = None
    if grids is not None and step_minutes == 1:
        cycle_table = cycle_performance(predictions, grids)[0].set_index("house")
    for house, frame in predictions.items():
        result = evaluate(frame["actual_w"], frame["predicted_w"], float(frame["threshold_w"].iloc[0]), step_minutes)
        extra = {} if cycle_table is None else {"cycle_f1": float(cycle_table.loc[house, "cycle_f1"]), "cycle_recall": float(cycle_table.loc[house, "cycle_recall"])}
        rows.append({"experiment": name, "label": label, "house": house, **result, **extra})
    pooled = pd.concat(predictions.values(), ignore_index=True)
    result = evaluate(pooled["actual_w"], pooled["predicted_w"], 0.0, step_minutes)
    result.update(on_off(pooled["actual_w"].to_numpy() > ON_W, pooled["predicted_w"].to_numpy() > pooled["threshold_w"].to_numpy()))
    extra = {} if cycle_table is None else {"cycle_f1": float(cycle_table.loc["pooled", "cycle_f1"]), "cycle_recall": float(cycle_table.loc["pooled", "cycle_recall"])}
    rows.append({"experiment": name, "label": label, "house": "pooled", **result, **extra})
    return rows


def compare_on_common_minutes(primary: dict[int, pd.DataFrame], variant: dict[int, pd.DataFrame]) -> list[dict]:
    """Score a variant on the primary model's minutes only, to isolate the training effect."""
    rows = []
    for house, frame in primary.items():
        joined = frame[["grid_pos", "actual_w"]].merge(variant[house][["grid_pos", "predicted_w", "threshold_w"]], on="grid_pos", how="inner")
        result = evaluate(joined["actual_w"], joined["predicted_w"], float(joined["threshold_w"].iloc[0]))
        rows.append({"experiment": "no_frozen_exclusion", "label": "Scored on the primary model's minutes", "house": house, **result})
    return rows


def resolution_tables(tables: dict[int, pd.DataFrame], step_minutes: int) -> dict[int, pd.DataFrame]:
    return {house: downsample(table, step_minutes) for house, table in tables.items()}


def aggregate_primary_to_interval(primary: dict[int, pd.DataFrame], step_minutes: int) -> dict[int, pd.DataFrame]:
    """Average 1-minute predictions to an interval, to compare with a model trained at that interval."""
    out = {}
    for house, frame in primary.items():
        seconds = (frame["official_minute"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
        bins = seconds.to_numpy() // (step_minutes * 60)
        grouped = frame.groupby(bins, sort=True)
        means = grouped[["actual_w", "predicted_w"]].mean()
        counts = grouped.size()
        means = means[counts >= int(np.ceil(0.8 * step_minutes))]
        out[house] = pd.DataFrame({"grid_pos": means.index.to_numpy(), "actual_w": means["actual_w"].to_numpy(), "predicted_w": means["predicted_w"].to_numpy(),
                                   "threshold_w": float(frame["threshold_w"].iloc[0]), "issues_any": False})
    return out


def signature_visibility(tables: dict[int, pd.DataFrame], steps=(1, 15, 30)) -> pd.DataFrame:
    """How visible is a wash in the aggregate as readings get coarser?

    Compares each cycle's largest interval of machine power with the household's ordinary
    interval-to-interval movement outside washing. A cycle counts as visible only if its
    power stands above the 90th percentile of that background movement.
    """
    rows = []
    for house, table in tables.items():
        cycles = detect_cycles(table["washing_machine_w"].where(table["valid"]).to_numpy(dtype="float64"), DEFAULT_PARAMS)
        full = cycles[cycles["complete"]]
        seconds = (table["official_minute"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
        for step in steps:
            coarse = table if step == 1 else downsample(table, step)
            power = coarse["washing_machine_w"].where(coarse["valid"]).to_numpy(dtype="float64")
            aggregate = coarse["aggregate_w"].where(coarse["valid"]).to_numpy(dtype="float64")
            moves = np.abs(np.diff(aggregate))
            quiet = (np.nan_to_num(power[1:], nan=0) <= ON_W) & (np.nan_to_num(power[:-1], nan=0) <= ON_W)
            background = float(np.nanpercentile(moves[quiet], 90))
            first_second = int(seconds.iloc[0])
            first_bin = first_second // (step * 60)
            peaks = []
            for _, cycle in full.iterrows():
                lo = (first_second + int(cycle["start_pos"]) * 60) // (step * 60) - first_bin
                hi = (first_second + int(cycle["end_pos"]) * 60) // (step * 60) - first_bin
                segment = power[lo:hi + 1]
                if len(segment) and np.isfinite(segment).any():
                    peaks.append(float(np.nanmax(segment)))
            peaks = np.array(peaks)
            rows.append({"house": house, "interval_minutes": step, "complete_cycles": len(peaks), "median_peak_machine_w": round(float(np.median(peaks)), 0),
                         "background_movement_p90_w": round(background, 0), "cycles_above_background_pct": round(float((peaks > background).mean() * 100), 1)})
    return pd.DataFrame(rows)
