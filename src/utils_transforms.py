# INSERIR este arquivo como completo em ~/nasa/src/utils_transforms.py
"""Utility transformers for the exoplanet tabular pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class ColumnAlignerConfig:
    """Configuration for :class:`ColumnAligner`.

    Attributes
    ----------
    columns:
        Optional list of column names that should be enforced during
        transformation. When ``None`` the columns seen during fitting are used.
    fill_value:
        Value used for columns that are missing in the input during
        transformation.
    """

    columns: Sequence[str] | None = None
    fill_value: float = np.nan


class ColumnAligner(BaseEstimator, TransformerMixin):
    """Ensure that tabular inputs have a fixed and ordered set of columns."""

    def __init__(self, columns: Iterable[str] | None = None, fill_value: float | int | str | None = np.nan) -> None:
        self.config = ColumnAlignerConfig(columns=list(columns) if columns is not None else None, fill_value=fill_value)  # type: ignore[arg-type]
        self.columns_: List[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> "ColumnAligner":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ColumnAligner expects a pandas DataFrame during fit.")
        if self.config.columns is not None:
            self.columns_ = list(self.config.columns)
        else:
            self.columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns_ is None:
            raise RuntimeError("ColumnAligner must be fitted before calling transform.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ColumnAligner expects a pandas DataFrame as input.")
        aligned = X.copy()
        missing = [col for col in self.columns_ if col not in aligned.columns]
        if missing:
            for col in missing:
                aligned[col] = self.config.fill_value
        aligned = aligned[self.columns_]
        return aligned

    def get_feature_names_out(self) -> np.ndarray:
        if self.columns_ is None:
            raise RuntimeError("ColumnAligner must be fitted before requesting feature names.")
        return np.asarray(self.columns_, dtype=object)
