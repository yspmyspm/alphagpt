from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc


@dataclass(frozen=True)
class FeatherColumnSource:
    """Read Feather data by schema and selected columns only."""

    default_data_path: str | None
    time_column: str

    def resolve_data_path(self, data_path: str | None = None) -> str:
        resolved = data_path or self.default_data_path
        if resolved is None:
            raise ValueError("No data path provided and no default_data_path configured")
        return str(Path(resolved).expanduser().resolve())

    def list_columns(self, data_path: str | None = None) -> list[str]:
        resolved_path = self.resolve_data_path(data_path)
        try:
            with pa.memory_map(resolved_path, "r") as source:
                reader = ipc.open_file(source)
                return list(reader.schema.names)
        except Exception:
            table = feather.read_table(resolved_path)
            return list(table.schema.names)

    def list_features(self, data_path: str | None = None) -> list[str]:
        return sorted(
            name
            for name in self.list_columns(data_path)
            if name != self.time_column
        )

    def load_frame(
        self,
        feature_names: Sequence[str],
        data_path: str | None = None,
    ) -> pd.DataFrame:
        resolved_path = self.resolve_data_path(data_path)
        requested_features = [str(name) for name in feature_names]
        read_columns = list(dict.fromkeys([self.time_column] + requested_features))
        try:
            frame = feather.read_table(resolved_path, columns=read_columns).to_pandas()
        except Exception as exc:
            raise ValueError(
                f"Failed to read requested columns {read_columns} from {resolved_path}: {type(exc).__name__}"
            ) from exc

        if frame.index.name == self.time_column:
            frame.sort_index(inplace=True)
            return frame

        if self.time_column not in frame.columns:
            raise ValueError(f"Time column {self.time_column!r} not found in {resolved_path}")

        frame = frame.set_index(self.time_column)
        frame.sort_index(inplace=True)
        return frame
