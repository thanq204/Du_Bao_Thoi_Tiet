from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


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
    plt.figure(figsize=(8, 4.5))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_prediction_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    timestamps: np.ndarray | None = None,
) -> None:
    plt.figure(figsize=(10, 4.8))
    if timestamps is None:
        x_axis = np.arange(len(y_true))
        plt.xlabel("Time step")
    else:
        x_axis = timestamps
        plt.xlabel("Timestamp")

    plt.plot(x_axis, y_true, label="Actual", linewidth=2)
    plt.plot(x_axis, y_pred, label="Predicted", linewidth=2)
    plt.ylabel("Value")
    plt.title("Weather forecast on test set")
    plt.legend()
    if timestamps is not None:
        plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


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
