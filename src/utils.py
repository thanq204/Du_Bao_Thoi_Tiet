from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str | None = None) -> torch.device:
    if preferred and preferred != "auto":
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": mae, "rmse": rmse}


def save_json(data: dict[str, object], output_path: Path) -> None:
    output_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def save_loss_plot(train_losses: list[float], val_losses: list[float], output_path: Path) -> None:
    figure = Figure(figsize=(8, 4.5), layout="tight")
    axes = figure.subplots()
    epochs = range(1, len(train_losses) + 1)
    axes.plot(epochs, train_losses, label="Train loss")
    axes.plot(epochs, val_losses, label="Validation loss")
    axes.set(xlabel="Epoch", ylabel="MSE loss", title="Training history")
    axes.legend()
    figure.savefig(output_path, dpi=150)


def save_prediction_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    timestamps: np.ndarray | None = None,
) -> None:
    figure = Figure(figsize=(10, 4.8), layout="tight")
    axes = figure.subplots()
    if timestamps is None:
        x_axis = np.arange(len(y_true))
        axes.set_xlabel("Time step")
    else:
        x_axis = timestamps
        axes.set_xlabel("Timestamp")
        axes.tick_params(axis="x", labelrotation=30)

    axes.plot(x_axis, y_true, label="Actual", linewidth=2)
    axes.plot(x_axis, y_pred, label="Predicted", linewidth=2)
    axes.set(ylabel="Value", title="Weather forecast on test set")
    axes.legend()
    figure.savefig(output_path, dpi=150)


def save_predictions_csv(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "actual", "predicted", "error"])
        for timestamp, actual, predicted in zip(timestamps, y_true, y_pred):
            writer.writerow(
                [
                    timestamp,
                    f"{float(actual):.4f}",
                    f"{float(predicted):.4f}",
                    f"{float(predicted - actual):.4f}",
                ]
            )
