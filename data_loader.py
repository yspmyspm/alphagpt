import os
import pandas as pd

from config import ModelConfig
from factors import FeatureEngineer


class AlphaDataLoader:
    """
    从本地 feather 加载数据。
    特征和 returns 均保留原始时间索引，不在加载时对齐（避免 rolling 前段 nan）。
    评估时再对齐。
    """

    def __init__(self):
        self.features = None  # List[pd.Series]，按 feature_cols 顺序
        self.raw_cache = None  # Dict[str, pd.Series]
        self.returns = None  # pd.Series，带时间索引

    def _load_feather(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feather not found: {path}")
        df = pd.read_feather(path)
        if "_time" in df.columns:
            df = df.set_index("_time")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    def load_data(self):
        # 1. 加载特征数据，保留原始 index
        print(f"Loading data from feather: {ModelConfig.FEATHER_PATH}")
        df = self._load_feather(ModelConfig.FEATHER_PATH)

        def col_to_series(col):
            return df[col].astype(float)

        if ModelConfig.FEATURE_COLUMNS is None:
            feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        else:
            feature_cols = list(ModelConfig.FEATURE_COLUMNS)
            missing = set(feature_cols) - set(df.columns)
            if missing:
                raise ValueError(f"Missing feature columns in feather: {missing}")

        if not feature_cols:
            raise ValueError("No feature columns found.")

        self.raw_cache = {
            c: col_to_series(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        }
        self.features = [self.raw_cache[c] for c in feature_cols]

        FeatureEngineer.INPUT_DIM = len(feature_cols)
        ModelConfig.INPUT_DIM = len(feature_cols)

        # 2. 加载 returns，保留原始 index
        path = os.path.join(ModelConfig.RETURNS_DIR, ModelConfig.RETURNS_FILENAME)
        df_ret = self._load_feather(path)
        col = ModelConfig.RETURNS_COLUMN
        if col not in df_ret.columns:
            raise ValueError(f"Column '{col}' not found in returns file.")
        self.returns = df_ret[col].astype(float)

        print(
            f"Data Ready. Features: {len(self.features)} cols ({len(df)} rows), "
            f"returns: {len(self.returns)} rows"
        )
