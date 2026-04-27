from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None

    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def interpolate_missing(values: np.ndarray) -> np.ndarray:
    filled = values.copy().astype(np.float32)
    row_indices = np.arange(len(filled))

    for column_index in range(filled.shape[1]):
        column = filled[:, column_index]
        valid_mask = ~np.isnan(column)
        if not valid_mask.any():
            raise ValueError(f"Column at index {column_index} contains only missing values.")
        filled[:, column_index] = np.interp(row_indices, row_indices[valid_mask], column[valid_mask])

    return filled


def add_time_features(timestamps: np.ndarray) -> tuple[list[str], np.ndarray]:
    hours = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps], dtype=np.float32)
    day_of_year = np.array([ts.timetuple().tm_yday for ts in timestamps], dtype=np.float32)

    derived_columns = ["hour_sin", "hour_cos", "dayofyear_sin", "dayofyear_cos"]
    derived_values = np.column_stack(
        [
            np.sin(2 * np.pi * hours / 24.0),
            np.cos(2 * np.pi * hours / 24.0),
            np.sin(2 * np.pi * day_of_year / 365.25),
            np.cos(2 * np.pi * day_of_year / 365.25),
        ]
    ).astype(np.float32)
    return derived_columns, derived_values


@dataclass
class WeatherTable:
    timestamps: np.ndarray
    values: np.ndarray
    columns: list[str]

    def __len__(self) -> int:
        return len(self.timestamps)

    def subset(self, start: int, end: int) -> "WeatherTable":
        return WeatherTable(
            timestamps=self.timestamps[start:end].copy(),
            values=self.values[start:end].copy(),
            columns=list(self.columns),
        )

    def matrix(self, selected_columns: list[str]) -> np.ndarray:
        indices = [self.columns.index(column) for column in selected_columns]
        return self.values[:, indices]


def split_table(table: WeatherTable, train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple[WeatherTable, WeatherTable, WeatherTable]:
    train_end = int(len(table) * train_ratio)
    val_end = train_end + int(len(table) * val_ratio)
    return table.subset(0, train_end), table.subset(train_end, val_end), table.subset(val_end, len(table))


@dataclass
class ArrayStandardScaler:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "ArrayStandardScaler":
        self.mean = values.mean(axis=0, keepdims=True).astype(np.float32)
        std = values.std(axis=0, keepdims=True).astype(np.float32)
        self.std = np.where(std > 1e-8, std, 1.0).astype(np.float32)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before calling transform().")
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before calling inverse_transform().")
        return (values * self.std + self.mean).astype(np.float32)


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.x = torch.from_numpy(features.astype(np.float32))
        self.y = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def parse_feature_columns(feature_columns_arg: str) -> list[str] | None:
    if not feature_columns_arg.strip():
        return None
    return [column.strip() for column in feature_columns_arg.split(",") if column.strip()]


def load_weather_table(
    csv_path: Path,
    time_column: str,
    target_column: str,
    feature_columns: list[str] | None = None,
    include_time_features: bool = True,
) -> tuple[WeatherTable, list[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required_columns = {time_column, target_column}
        missing_required = required_columns.difference(fieldnames)
        if missing_required:
            missing_names = ", ".join(sorted(missing_required))
            raise ValueError(f"Missing required columns in CSV: {missing_names}")

        raw_columns = [column for column in fieldnames if column != time_column]
        parsed_rows: list[tuple[datetime, dict[str, str]]] = []
        for row in reader:
            timestamp = parse_timestamp(row.get(time_column, ""))
            if timestamp is not None:
                parsed_rows.append((timestamp, row))

    parsed_rows.sort(key=lambda item: item[0])

    deduplicated_rows: dict[datetime, dict[str, str]] = {}
    for timestamp, row in parsed_rows:
        deduplicated_rows[timestamp] = row

    ordered_timestamps = np.array(sorted(deduplicated_rows.keys()), dtype=object)
    if len(ordered_timestamps) == 0:
        raise ValueError("No valid timestamps were found in the CSV file.")

    raw_values = []
    for timestamp in ordered_timestamps:
        row = deduplicated_rows[timestamp]
        numeric_row = []
        for column in raw_columns:
            raw_value = (row.get(column) or "").strip()
            try:
                numeric_row.append(float(raw_value))
            except ValueError:
                numeric_row.append(np.nan)
        raw_values.append(numeric_row)

    values = np.array(raw_values, dtype=np.float32)
    values = interpolate_missing(values)
    all_columns = list(raw_columns)

    if include_time_features:
        derived_columns, derived_values = add_time_features(ordered_timestamps)
        values = np.concatenate([values, derived_values], axis=1)
        all_columns.extend(derived_columns)
    else:
        derived_columns = []

    if feature_columns is None:
        feature_columns = [target_column] + [column for column in raw_columns if column != target_column]
    else:
        feature_columns = list(feature_columns)
        if target_column not in feature_columns:
            feature_columns = [target_column] + feature_columns

    if include_time_features:
        for column in derived_columns:
            if column not in feature_columns:
                feature_columns.append(column)

    missing_features = [column for column in feature_columns if column not in all_columns]
    if missing_features:
        missing_names = ", ".join(missing_features)
        raise ValueError(f"Missing feature columns in CSV: {missing_names}")

    return WeatherTable(timestamps=ordered_timestamps, values=values, columns=all_columns), feature_columns


def make_windowed_arrays(
    features: np.ndarray,
    targets: np.ndarray,
    timestamps: np.ndarray,
    sequence_length: int,
    forecast_horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(features) <= sequence_length + forecast_horizon - 1:
        raise ValueError("Not enough rows to create sequences with the selected sequence_length and forecast_horizon.")

    window_features = []
    window_targets = []
    window_timestamps = []

    last_start = len(features) - sequence_length - forecast_horizon + 1
    for start in range(last_start):
        target_index = start + sequence_length + forecast_horizon - 1
        window_features.append(features[start : start + sequence_length])
        window_targets.append(targets[target_index])
        window_timestamps.append(timestamps[target_index])

    return (
        np.stack(window_features).astype(np.float32),
        np.array(window_targets, dtype=np.float32).reshape(-1, 1),
        np.array(window_timestamps, dtype=object),
    )
