"""Base scorer interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class ScoreResult:
    score: Optional[float]
    metrics: Dict[str, Any] = field(default_factory=dict)


class AbstractSeriesScorer(ABC):

    @abstractmethod
    def score_series(
        self,
        series: pd.Series,
        returns: pd.Series,
        *,
        subset_index: Optional[pd.Index] = None,
    ) -> ScoreResult:
        ...
