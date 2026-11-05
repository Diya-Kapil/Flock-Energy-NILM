"""Small display helpers for the notebooks, which only show results the pipeline produced."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from IPython.display import Image, display

from src.config import ROOT
from src.io_utils import read_json


def table(relative_path: str, columns: list[str] | None = None, query: str | None = None, transpose: bool = False) -> None:
    """Show a saved CSV. Fails clearly if the pipeline has not produced it yet."""
    frame = pd.read_csv(ROOT / relative_path)
    if query:
        frame = frame.query(query)
    if columns:
        frame = frame[columns]
    display(frame.set_index(frame.columns[0]).T if transpose else frame)


def plot(relative_path: str, width: int = 1000) -> None:
    """Show a saved plot."""
    path = Path(ROOT / relative_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.pipeline")
    display(Image(filename=str(path), width=width))


def report(relative_path: str) -> dict:
    """Load a saved JSON report."""
    return read_json(ROOT / relative_path)
