"""Trainer interface and fit result container."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

from pool_state import PoolEntry
import numpy as np
import pandas as pd


@dataclass
class EnsembleFitResult:
    importance: np.ndarray
    score: Optional[float]
    prediction: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


class AbstractEnsembleTrainer(ABC):

    @abstractmethod
    def fit_and_predict(
        self,
        factors: List[PoolEntry],
        returns: pd.Series,
        data_idx: pd.Index,
        *,
        scorer: Any,
    ) -> EnsembleFitResult:
        """Fit the ensemble model and return importance, score, and the combo prediction.

        The returned ``prediction`` is computed on ``data_idx`` using whatever
        normalisation / model the trainer applies.  Callers must not assume a
        linear ``X @ w`` structure.
        """
        ...

    @abstractmethod
    def diversity_score(
        self,
        candidate: pd.Series,
        pool_factors: List[Any],
        data_idx: pd.Index,
    ) -> float:
        ...
