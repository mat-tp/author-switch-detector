"""
feature_vector.py — FeatureVector class.

Wraps a feature array together with the names of each dimension.
"""

from typing import List

import numpy as np


class FeatureVector:
    """
    Holds a feature array plus the name of each dimension.

    Attributes
    ----------
    values : np.ndarray  — float32, shape [n_features] or [n_sentences, n_features]
    names  : List[str]   — one name per feature column
    """

    def __init__(self, values: np.ndarray, names: List[str]):
        self.values: np.ndarray = np.asarray(values, dtype=np.float32)
        self.names:  List[str]  = list(names)

    def __len__(self) -> int:
        """Number of feature names (columns)."""
        return len(self.names)

    def __repr__(self) -> str:
        return f"FeatureVector(shape={self.values.shape}, n_features={len(self.names)})"
