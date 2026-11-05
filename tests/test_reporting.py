"""The generated documents: decision rules, and agreement with the saved tables."""
from __future__ import annotations

import re

import pandas as pd
import pytest

from src import config, reporting


def experiments(primary_mae: float, primary_f1: float, two_mae: float, two_f1: float) -> pd.DataFrame:
    return pd.DataFrame([{"experiment": "primary", "house": "pooled", "mae_w": primary_mae, "f1": primary_f1},
                         {"experiment": "two_stage", "house": "pooled", "mae_w": two_mae, "f1": two_f1}])


class TestTwoStageRule:
    def test_adopted_only_with_enough_mae_gain_and_f1_no_lower(self):
        assert reporting.two_stage_verdict(experiments(30, 0.4, 26.9, 0.4))["adopted"]

    def test_small_gain_is_not_adopted(self):
        assert not reporting.two_stage_verdict(experiments(30, 0.4, 28, 0.5))["adopted"]

    def test_lower_f1_is_not_adopted(self):
        assert not reporting.two_stage_verdict(experiments(30, 0.4, 20, 0.39))["adopted"]


def test_markdown_tables_keep_house_order():
    metrics = pd.DataFrame([{"house": h, "model": m, "subset": "all_valid", "minutes": 10, "mae_w": 1.0, "rmse_w": 1.0, "r2": 0.1, "mae_active_w": 1.0, "precision": 0.5, "recall": 0.5, "f1": 0.5, "energy_error_pct": 1.0}
                            for h in (1, "pooled") for m in ("random_forest", "always_zero")])
    text = reporting.per_house_table(metrics)
    assert text.index("| 1 |") < text.index("| Pooled |")


results = config.ROOT / "RESULTS.md"
recommendation = config.ROOT / "Recommendation.md"
outputs_exist = results.exists() and results.stat().st_size > 0 and (config.MODEL_REPORTS_DIR / "metrics.csv").exists()


@pytest.mark.skipif(not outputs_exist, reason="Run python -m src.pipeline first.")
class TestGeneratedDocuments:
    def test_results_quote_the_saved_pooled_metrics(self):
        metrics = pd.read_csv(config.MODEL_REPORTS_DIR / "metrics.csv")
        text = results.read_text()
        forest = reporting.pooled_row(metrics, "random_forest")
        zero = reporting.pooled_row(metrics, "always_zero")
        assert f"{forest['mae_w']:.1f} W against {zero['mae_w']:.1f} W" in text

    def test_recommendation_quotes_the_saved_saving(self):
        opportunity = pd.read_csv(config.RECOMMENDATION_REPORTS_DIR / "opportunity.csv")
        assert f"{opportunity['saving_cap_at_median_kwh_per_year'].mean():.0f} kWh a year" in recommendation.read_text()

    @pytest.mark.parametrize("document", ["RESULTS.md", "Recommendation.md", "README.md", "DATA.md"])
    def test_every_local_link_and_image_exists(self, document):
        text = (config.ROOT / document).read_text()
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith("http") or target.startswith("#"):
                continue
            assert (config.ROOT / target.split("#")[0]).exists(), target

    @pytest.mark.parametrize("document", ["README.md", "DATA.md", "RESULTS.md", "Recommendation.md"])
    def test_required_documents_are_not_empty(self, document):
        assert (config.ROOT / document).stat().st_size > 500

    def test_notebooks_exist_with_outputs(self):
        for name in ("01_data_cleaning", "02_washing_machine_analysis", "03_disaggregation_model", "04_recommendations"):
            path = config.NOTEBOOKS_DIR / f"{name}.ipynb"
            assert path.exists()
            assert '"outputs": [\n' in path.read_text() or '"output_type"' in path.read_text()
