"""Notebooks display results only, so they must not hold pipeline logic."""
from __future__ import annotations

import pytest

from src import build_notebooks

FORBIDDEN = ("RandomForest", ".fit(", "groupby(\"Unix\"", "make_features", "detect_cycles", "read_raw_part", "py7zr", "to_parquet", "to_csv")


@pytest.mark.parametrize("name,cells", build_notebooks.NOTEBOOKS.items())
def test_notebook_cells_contain_no_pipeline_logic(name, cells):
    for cell in cells():
        if cell["cell_type"] == "code":
            for word in FORBIDDEN:
                assert word not in cell["source"], f"{name}: {word}"


def test_there_are_four_notebooks_with_the_required_names():
    assert list(build_notebooks.NOTEBOOKS) == ["01_data_cleaning.ipynb", "02_washing_machine_analysis.ipynb",
                                               "03_disaggregation_model.ipynb", "04_recommendations.ipynb"]
