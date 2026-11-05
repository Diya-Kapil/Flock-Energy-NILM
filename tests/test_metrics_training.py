"""Metrics, training-row sampling and household separation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import dataset, evaluate as ev, metrics, train
from src.models.baselines import always_zero

SMALL = train.ForestSettings(n_estimators=8, min_samples_leaf=2, rows_per_house=400)


class TestEvaluate:
    def test_perfect_prediction(self):
        y = np.array([0.0, 0.0, 100.0, 2000.0])
        out = metrics.evaluate(y, y, threshold_w=20)
        assert out["mae_w"] == 0
        assert out["rmse_w"] == 0
        assert out["r2"] == 1
        assert out["f1"] == 1
        assert out["energy_error_pct"] == 0

    def test_known_errors(self):
        y, p = np.array([0.0, 100.0]), np.array([10.0, 70.0])
        out = metrics.evaluate(y, p, threshold_w=20)
        assert out["mae_w"] == 20
        assert out["rmse_w"] == pytest.approx(np.sqrt((100 + 900) / 2))
        assert out["mae_active_w"] == 30
        assert out["energy_error_pct"] == -20

    def test_always_zero_baseline(self):
        y = np.array([0.0, 0.0, 0.0, 400.0])
        out = metrics.evaluate(y, always_zero(4), threshold_w=20)
        assert out["mae_w"] == 100
        assert out["mae_active_w"] == 400
        assert (out["precision"], out["recall"], out["f1"]) == (0, 0, 0)
        assert out["energy_error_pct"] == -100
        assert out["r2"] < 0

    def test_energy_uses_the_interval(self):
        y = np.full(2, 6000.0)
        assert metrics.evaluate(y, y, 20)["actual_kwh"] == pytest.approx(0.2)
        assert metrics.evaluate(y, y, 20, step_minutes=30)["actual_kwh"] == pytest.approx(6.0)

    def test_on_off_counts(self):
        out = metrics.on_off(np.array([True, True, False, False]), np.array([True, False, True, False]))
        assert (out["tp"], out["fp"], out["fn"]) == (1, 1, 1)
        assert out["f1"] == pytest.approx(0.5)


def test_best_threshold_finds_the_separating_value():
    y = np.array([0, 0, 0, 500, 500], dtype=float)
    p = np.array([10, 40, 60, 150, 300], dtype=float)
    threshold, f1 = metrics.best_threshold(y, p)
    assert f1 == 1
    assert 60 <= threshold < 150


class TestMatchCycles:
    @staticmethod
    def cycles(*spans) -> pd.DataFrame:
        return pd.DataFrame(spans, columns=["start_pos", "end_pos"])

    def test_overlap_counts_as_a_match(self):
        out = metrics.match_cycles(self.cycles((100, 159)), self.cycles((110, 170)))
        assert (out["matched"], out["cycle_precision"], out["cycle_recall"]) == (1, 1, 1)

    def test_small_overlap_does_not_count(self):
        out = metrics.match_cycles(self.cycles((100, 159)), self.cycles((155, 215)))
        assert out["matched"] == 0
        assert out["cycle_f1"] == 0

    def test_one_prediction_cannot_match_two_cycles(self):
        out = metrics.match_cycles(self.cycles((100, 130), (140, 170)), self.cycles((100, 170)))
        assert out["matched"] == 1
        assert out["cycle_recall"] == 0.5

    def test_no_predictions(self):
        out = metrics.match_cycles(self.cycles((100, 130)), pd.DataFrame(columns=["start_pos", "end_pos"]))
        assert out["matched"] == 0
        assert out["matched_flags"] == [False]


class TestSampling:
    def test_natural_sampling_keeps_the_natural_mix(self):
        y = np.r_[np.zeros(9000), np.full(1000, 500.0)]
        chosen = dataset.sample_rows(y, 2000, None, np.random.default_rng(0))
        assert len(chosen) == 2000
        assert (y[chosen] > 20).mean() == pytest.approx(0.1, abs=0.03)

    def test_ratio_is_respected(self):
        y = np.r_[np.zeros(9000), np.full(1000, 500.0)]
        chosen = dataset.sample_rows(y, 1000, 4.0, np.random.default_rng(0))
        assert (y[chosen] > 20).sum() == 200
        assert (y[chosen] <= 20).sum() == 800

    def test_never_takes_more_than_exists(self):
        y = np.r_[np.zeros(100), np.full(10, 500.0)]
        chosen = dataset.sample_rows(y, 1000, 1.0, np.random.default_rng(0))
        assert (y[chosen] > 20).sum() == 10
        assert len(np.unique(chosen)) == len(chosen)

    def test_same_seed_same_rows(self):
        y = np.r_[np.zeros(500), np.full(100, 500.0)]
        first = dataset.sample_rows(y, 200, 1.0, np.random.default_rng(5))
        second = dataset.sample_rows(y, 200, 1.0, np.random.default_rng(5))
        assert (first == second).all()


def synthetic_house(house: int, seed: int, n: int = 6000) -> dataset.HouseData:
    """Aggregate with noise, plus a machine pattern the features can see."""
    rng = np.random.default_rng(seed)
    machine = np.zeros(n)
    for start in rng.integers(0, n - 100, size=n // 400):
        machine[start:start + 20] = 2000
        machine[start + 20:start + 80] = 150
    a = 300 + rng.normal(0, 20, n) + machine
    table = pd.DataFrame({"aggregate_w": a.astype("float32"), "washing_machine_w": machine.astype("float32"), "valid": True, "missing": False})
    return dataset.build_house(table, house)


@pytest.fixture(scope="module")
def houses() -> list[dataset.HouseData]:
    return [synthetic_house(h, 100 + h) for h in (1, 2, 3)]


class TestHouseSeparation:
    def test_training_matrix_uses_only_the_houses_it_is_given(self, houses):
        marker = houses[0].X.copy()
        marker["aggregate"] = -1.0
        polluted = dataset.HouseData(house=1, positions=houses[0].positions, X=marker, y=houses[0].y)
        X, _ = dataset.training_matrix(houses[1:], 500, None, seed=1)
        assert not (X["aggregate"] == -1).any()
        X2, _ = dataset.training_matrix([polluted, *houses[1:]], 500, None, seed=1)
        assert (X2["aggregate"] == -1).any()

    def test_fold_refuses_a_held_out_house_in_training(self, houses):
        with pytest.raises(AssertionError):
            train.run_fold(houses[0], houses, SMALL, None, seed=1)

    def test_selection_never_sees_the_held_out_house(self, houses, monkeypatch):
        seen = []
        original = dataset.training_matrix

        def spy(chosen, *args, **kwargs):
            seen.extend(h.house for h in chosen)
            return original(chosen, *args, **kwargs)

        monkeypatch.setattr(train, "training_matrix", spy)
        train.select_on_training_houses(houses[1:], seed=1, options=(None,), settings=SMALL)
        assert 1 not in seen
        assert set(seen) == {2, 3}


class TestForest:
    def test_learns_an_easy_pattern_better_than_zero(self, houses):
        _, predicted, info = train.run_fold(houses[0], houses[1:], SMALL, 1.0, seed=1)
        y = houses[0].y
        assert np.abs(predicted - y).mean() < np.abs(y).mean()
        assert info["training_rows"] > 0

    def test_same_seed_gives_identical_predictions(self, houses):
        _, first, _ = train.run_fold(houses[0], houses[1:], SMALL, 1.0, seed=1)
        _, second, _ = train.run_fold(houses[0], houses[1:], SMALL, 1.0, seed=1)
        assert np.array_equal(first, second)

    def test_handles_missing_feature_values(self, houses):
        X = houses[1].X.copy()
        X.iloc[:50, 1:5] = np.nan
        forest = train.fit_forest(X, houses[1].y, SMALL, 1)
        assert np.isfinite(train.predict(forest, X)).all()

    def test_selection_picks_a_threshold_from_the_grid(self, houses):
        table, chosen, threshold = train.select_on_training_houses(houses, seed=1, options=(None, 1.0), settings=SMALL)
        assert len(table) == 2
        assert threshold in metrics.THRESHOLD_GRID_W
        assert chosen in (None, 1.0)


def predictions_frame(actual, predicted, threshold=50.0, issues=None) -> pd.DataFrame:
    n = len(actual)
    return pd.DataFrame({"grid_pos": np.arange(n), "actual_w": np.asarray(actual, dtype="float64"), "predicted_w": np.asarray(predicted, dtype="float64"),
                         "threshold_w": threshold, "issues_any": np.zeros(n, bool) if issues is None else issues,
                         "aggregate_w": 300.0, "uk_local_minute": pd.date_range("2014-05-07", periods=n, freq="min")})


class TestEvaluation:
    def test_tables_have_house_pooled_and_both_subsets(self):
        frame = predictions_frame([0, 0, 500, 500], [0, 60, 400, 500])
        table = ev.metric_tables({1: frame, 2: frame})
        assert set(table["house"].astype(str)) == {"1", "2", "pooled"}
        assert set(table["subset"]) == {"all_valid", "excluding_issues"}
        assert set(table["model"]) == {"random_forest", "always_zero"}

    def test_issues_rows_leave_only_the_secondary_subset(self):
        frame = predictions_frame([0, 0, 500, 500], [0, 0, 500, 0], issues=np.array([False, False, False, True]))
        table = ev.metric_tables({1: frame}).query("house == 1 and model == 'random_forest'").set_index("subset")
        assert table.loc["all_valid", "minutes"] == 4
        assert table.loc["excluding_issues", "minutes"] == 3
        assert table.loc["excluding_issues", "mae_w"] == 0
        assert table.loc["all_valid", "mae_w"] == 125

    def test_pooled_on_off_uses_each_folds_own_threshold(self):
        first = predictions_frame([0, 500], [60, 60], threshold=100.0)
        second = predictions_frame([0, 500], [60, 60], threshold=10.0)
        pooled = ev.metric_tables({1: first, 2: second}).query("house == 'pooled' and model == 'random_forest' and subset == 'all_valid'").iloc[0]
        assert (pooled["tp"], pooled["fp"], pooled["fn"]) == (1, 1, 1)

    def test_cycle_performance_finds_a_matching_cycle(self):
        actual = np.zeros(300)
        actual[100:160] = 2000
        frame = predictions_frame(actual, actual * 0.8)
        table, details = ev.cycle_performance({1: frame}, {1: 300})
        assert table.iloc[0]["matched"] == 1
        assert details["detected"].all()

    def test_false_alarm_episode_is_found(self):
        predicted = np.zeros(300)
        predicted[100:140] = 800
        episodes = ev.false_alarm_episodes(predictions_frame(np.zeros(300), predicted), 300)
        assert len(episodes) == 1

    def test_false_positive_sources_name_the_running_appliance(self):
        frame = predictions_frame([0, 0, 0], [80, 80, 0])
        tables = {1: pd.DataFrame({"other_appliance6_w": [2000.0, 0.0, 0.0]})}
        out = ev.false_positive_sources({1: frame}, tables, {1: {"other_appliance6_w": "Dishwasher"}}).set_index("running_appliance")
        assert out.loc["Dishwasher", "false_alarm_minutes"] == 1
        assert "No metered appliance at 100 W or more" in out.index

    def test_peak_share_known_answer(self):
        minutes = pd.date_range("2014-05-07 15:00", periods=120, freq="min")
        frame = pd.DataFrame({"uk_local_minute": minutes, "actual_w": 100.0})
        assert ev.peak_share(frame, "actual_w") == 50

    def test_macro_mean_averages_houses_not_minutes(self):
        small = predictions_frame([100.0], [0.0])
        large = predictions_frame([0.0] * 9, [0.0] * 9)
        table = ev.metric_tables({1: small, 2: large})
        macro = ev.macro_means(table).query("model == 'random_forest' and subset == 'all_valid'").iloc[0]
        assert macro["mae_w"] == 50


class TestExperiments:
    def test_two_stage_is_zero_where_classifier_says_off(self, houses):
        from src.models import experiments
        X, y = dataset.training_matrix(houses[1:], 400, 1.0, seed=1)
        classifier, regressor = experiments.fit_two_stage(X, y, SMALL, 1)
        predicted = experiments.two_stage_predict(classifier, regressor, houses[0].X, 0.5)
        assert (predicted >= 0).all()
        assert (predicted == 0).any()
        assert experiments.two_stage_predict(classifier, regressor, houses[0].X, 1.01).sum() == 0

    def test_probability_cutoff_comes_from_the_grid(self, houses):
        from src.models import experiments
        assert experiments.select_probability_cutoff(houses, 1.0, 1, settings=SMALL) in experiments.PROBABILITY_GRID

    def test_common_minutes_are_scored_on_the_primary_set(self):
        from src.models import experiments
        primary = {1: predictions_frame([0, 500], [0, 500])}
        variant = {1: predictions_frame([0, 500, 900], [0, 500, 0])}
        row = experiments.compare_on_common_minutes(primary, variant)[0]
        assert row["minutes"] == 2
        assert row["mae_w"] == 0

    def test_interval_average_needs_coverage(self):
        from src.models import experiments
        minutes = pd.date_range("2014-05-07 00:00", periods=45, freq="min")
        frame = predictions_frame(np.full(45, 60.0), np.full(45, 30.0)).assign(official_minute=minutes)
        out = experiments.aggregate_primary_to_interval({1: frame}, 30)[1]
        assert len(out) == 1
        assert out["actual_w"].iloc[0] == 60
