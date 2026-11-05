"""The comparison model: always predict that the washing machine is off."""
from __future__ import annotations

import numpy as np


def always_zero(n: int) -> np.ndarray:
    return np.zeros(n, dtype="float64")
