"""Write and execute the four notebooks. They display pipeline outputs and hold no pipeline logic."""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from src.config import NOTEBOOKS_DIR, ROOT

SETUP = """import os, sys
from pathlib import Path
root = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
os.chdir(root)
sys.path.insert(0, str(root))
import pandas as pd
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)
from src.notebook_helpers import plot, report, table"""


def md(text: str):
    return new_markdown_cell(text.strip())


def code(text: str):
    return new_code_cell(text.strip())


def cleaning() -> list:
    return [
        md("""# 1. Raw House 1 cleaning

Shows the outputs of `python -m src.pipeline --stage clean`. The logic lives in `src/cleaning/`, and this notebook only displays results. The reasoning for each decision is in [DATA.md](../DATA.md)."""),
        code(SETUP),
        md("## Row counts through cleaning"),
        code("""flow = report("outputs/reports/cleaning_report.json")["row_flow"]
pd.Series(flow, name="value").to_frame()"""),
        md("## Missing values per sensor\n\nNothing is filled or interpolated. Minutes with no reading stay missing."),
        code("""pd.DataFrame(report("outputs/reports/cleaning_report.json")["missing_values"][:-1])"""),
        md("## Outliers\n\nAppliance readings above 4000 W become missing. Aggregate readings above 23 kW are kept and flagged."),
        code("""pd.DataFrame(report("outputs/reports/cleaning_report.json")["outliers"]["appliance_spikes"]).drop(columns="commonest_values")"""),
        code("""pd.DataFrame(report("outputs/reports/cleaning_report.json")["outliers"]["aggregate_review"]["episodes"])"""),
        md("## Outages and frozen sensors"),
        code("""pd.DataFrame(report("outputs/reports/cleaning_report.json")["outages"]["buckets"])"""),
        code("""pd.DataFrame(report("outputs/reports/cleaning_report.json")["frozen"]["per_sensor"])"""),
        md("## Before and after"),
        code("""plot("outputs/plots/cleaning/before_after_aggregate.png")"""),
        code("""plot("outputs/plots/cleaning/before_after_appliance5.png")"""),
        code("""plot("outputs/plots/cleaning/coverage_by_sensor.png")"""),
        md("## Agreement with the official cleaned file\n\nCompared on the same clock. Agreement is close where both use the same raw part."),
        code("""periods = report("outputs/reports/reconciliation.json")["periods"]
pd.DataFrame([{"period": name, **row} for name, value in periods.items() for row in value["sensors"] if row["sensor"] in ("Aggregate", "Appliance5")])"""),
    ]


def analysis() -> list:
    return [
        md("""# 2. Washing machine analysis

Shows the outputs of `python -m src.pipeline --stage eda`, from official cleaned REFIT data for Houses 1, 2, 8, 10, 15 and 16. Frozen minutes are excluded. Hour and weekday results use the verified UK local clock."""),
        code(SETUP),
        md("## Data quality per household"),
        code("""table("outputs/reports/eda/data_quality.csv", transpose=True)"""),
        md("## Off and active time"),
        code("""table("outputs/reports/eda/power_states.csv", transpose=True)"""),
        md("## Cycle definition"),
        code("""pd.Series(report("outputs/reports/eda/cycle_parameters.json"), name="value").to_frame()"""),
        md("## Cycle summary per household"),
        code("""table("outputs/reports/eda/cycle_summary.csv", transpose=True)"""),
        md("## Representative days"),
        code("""plot("outputs/plots/eda/representative_days.png")"""),
        md("## Detected cycles"),
        code("""plot("outputs/plots/eda/detected_cycles.png")"""),
        md("## Hour of day and day of week"),
        code("""plot("outputs/plots/eda/hourly_distribution.png")"""),
        code("""plot("outputs/plots/eda/weekday_distribution.png")"""),
        md("## Cross-household comparison"),
        code("""plot("outputs/plots/eda/cross_household.png")"""),
        code("""plot("outputs/plots/eda/duration_energy.png")"""),
        md("## Sensitivity checks\n\nCycle counts under other detector settings, then the effect of frozen minutes and the clock correction."),
        code("""pd.read_csv("outputs/reports/eda/parameter_sensitivity.csv").pivot(index="setting", columns="house", values="cycles")"""),
        code("""table("outputs/reports/eda/frozen_sensitivity.csv", transpose=True)"""),
        code("""table("outputs/reports/eda/clock_sensitivity.csv", transpose=True)"""),
    ]


def model() -> list:
    return [
        md("""# 3. Disaggregation model

Shows the outputs of `python -m src.pipeline --stage train`, `evaluate` and `experiments`. Every metric comes from a house the model did not train on, across all of that house's valid minutes. Input is aggregate power only, using past readings only."""),
        code(SETUP),
        md("## Training choices\n\nThe sampling mix and on/off threshold were chosen by inner validation on training houses only."),
        code("""log = report("outputs/reports/model/training_log.json")
display(pd.Series(log["settings"], name="value").to_frame())
pd.DataFrame(log["folds"]).drop(columns="training_houses")"""),
        md("## Model against baseline, all valid minutes"),
        code("""columns = ["house", "model", "minutes", "mae_w", "rmse_w", "r2", "mae_active_w", "precision", "recall", "f1", "energy_error_pct"]
table("outputs/reports/model/metrics.csv", columns=columns, query="subset == 'all_valid'")"""),
        code("""plot("outputs/plots/model/model_vs_baseline.png")"""),
        md("## Actual and predicted power"),
        code("""plot("outputs/plots/model/actual_vs_predicted.png")"""),
        code("""plot("outputs/plots/model/predicted_against_actual.png")"""),
        md("## With and without Issues = 1"),
        code("""frame = pd.read_csv("outputs/reports/model/metrics.csv").query("model == 'random_forest'")
frame.pivot(index="house", columns="subset", values=["minutes", "mae_w", "mae_active_w", "f1", "energy_error_pct"])"""),
        md("## Cycle detection"),
        code("""table("outputs/reports/model/cycle_detection.csv")"""),
        code("""table("outputs/reports/model/cycle_recall_by_type.csv")"""),
        md("## Error analysis"),
        code("""table("outputs/reports/model/error_sources.csv")"""),
        code("""table("outputs/reports/model/error_by_aggregate_level.csv")"""),
        code("""table("outputs/reports/model/false_alarm_sources.csv")"""),
        code("""plot("outputs/plots/model/failure_cases.png")"""),
        code("""plot("outputs/plots/model/feature_importance.png")"""),
        md("## Household-level estimates a utility might use"),
        code("""table("outputs/reports/model/practical_estimates.csv")"""),
        md("## Secondary experiments\n\nNone of these replaces the primary result."),
        code("""columns = ["experiment", "label", "minutes", "mae_w", "rmse_w", "r2", "mae_active_w", "precision", "recall", "f1", "energy_error_pct", "cycle_f1"]
table("outputs/reports/model/experiments.csv", columns=columns, query="house == 'pooled'")"""),
    ]


def recommendations() -> list:
    return [
        md("""# 4. Recommendations

Shows the figures behind [Recommendation.md](../Recommendation.md), from `python -m src.pipeline --stage recommend`. Savings are scenarios built from each household's own cycles, not forecasts."""),
        code(SETUP),
        md("## The opportunity, per household"),
        code("""table("outputs/reports/recommendation/opportunity.csv", transpose=True)"""),
        code("""plot("outputs/plots/recommendation/saving_by_household.png")"""),
        md("## How hard the saving is to measure"),
        code("""table("outputs/reports/recommendation/measurement_power.csv")"""),
        md("## What 15-minute and 30-minute readings lose"),
        code("""table("outputs/reports/model/signature_visibility.csv")"""),
        code("""plot("outputs/plots/recommendation/resolution.png")"""),
        md("## Model reliability at household level"),
        code("""table("outputs/reports/model/practical_estimates.csv")"""),
    ]


NOTEBOOKS = {
    "01_data_cleaning.ipynb": cleaning,
    "02_washing_machine_analysis.ipynb": analysis,
    "03_disaggregation_model.ipynb": model,
    "04_recommendations.ipynb": recommendations,
}


def build_and_execute(folder: Path = NOTEBOOKS_DIR, root: Path = ROOT) -> list[Path]:
    """Write each notebook, run it from the project root, and save it with its outputs."""
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for name, cells in NOTEBOOKS.items():
        notebook = new_notebook(cells=cells(), metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
        NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
        path = folder / name
        nbformat.write(notebook, path)
        written.append(path)
    return written
