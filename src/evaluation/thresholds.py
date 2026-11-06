"""Threshold selection. Thresholds come ONLY from validation normal scores."""
from __future__ import annotations

import numpy as np

from .schema import Predictions


def percentile_threshold(val: Predictions, percentile: float = 95.0) -> float:
    """Percentile of validation normal scores (per category).

    Raises if the split is not "val" or is not purely normal, so the test set
    can never be used to pick a threshold by accident.
    """
    if val.split != "val":
        raise ValueError(f"threshold must be fit on split val, got {val.split!r}")
    if (val.labels != 0).any():
        raise ValueError("validation split must contain normal images only")
    if len(val.scores) == 0:
        raise ValueError("empty validation split")
    return float(np.percentile(val.scores, percentile))
