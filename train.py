from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import (
    ArrayStandardScaler,
    SequenceDataset,
    WeatherTable,
    load_weather_table,
    make_windowed_arrays,
    parse_feature_columns,
    split_table,
)
from src.models import RecurrentForecaster
from src.utils import (
    get_device,
    regression_metrics,
    save_json,
    save_loss_plot,
    save_prediction_plot,
    save_predictions_csv,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RNN-based weather forecasting model.")
    parser.add_argument("--data-path", type=Path, default=Path("data/weather_data.csv"))
    parser.add_argument("--time-column", type=str, default="timestamp")
    parser.add_argument("--target-column", type=str, default="temperature_2m")
    parser.add_argument("--feature-columns", type=str, default="")
    parser.add_argument("--no-time-features", action="store_true")
    parser.add_argument("--model", choices=["rnn", "lstm", "gru"], default="lstm")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--forecast-horizon", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    predictions = []
    targets = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            losses.append(loss.item())
            predictions.append(outputs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    predictions_array = np.concatenate(predictions, axis=0).reshape(-1, 1)
    targets_array = np.concatenate(targets, axis=0).reshape(-1, 1)
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return mean_loss, predictions_array, targets_array


def build_dataset(
    table: WeatherTable,
    feature_columns: list[str],
    target_column: str,
    feature_scaler: ArrayStandardScaler,
    target_scaler: ArrayStandardScaler,
    sequence_length: int,
    forecast_horizon: int,
) -> tuple[SequenceDataset, np.ndarray]:
    feature_values = feature_scaler.transform(table.matrix(feature_columns).astype(np.float32))
    target_values = target_scaler.transform(table.matrix([target_column]).astype(np.float32))
    timestamps = table.timestamps

    x, y, target_timestamps = make_windowed_arrays(
        features=feature_values,
        targets=target_values,
        timestamps=timestamps,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
    )
    return SequenceDataset(x, y), target_timestamps


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {args.data_path}. Add a CSV file or run scripts/fetch_weather_data.py first."
        )

    raw_feature_columns = parse_feature_columns(args.feature_columns)
    table, feature_columns = load_weather_table(
        csv_path=args.data_path,
        time_column=args.time_column,
        target_column=args.target_column,
        feature_columns=raw_feature_columns,
        include_time_features=not args.no_time_features,
    )

    train_table, val_table, test_table = split_table(table, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    feature_scaler = ArrayStandardScaler().fit(train_table.matrix(feature_columns).astype(np.float32))
    target_scaler = ArrayStandardScaler().fit(train_table.matrix([args.target_column]).astype(np.float32))

    train_dataset, _ = build_dataset(
        table=train_table,
        feature_columns=feature_columns,
        target_column=args.target_column,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
    )
    val_dataset, _ = build_dataset(
        table=val_table,
        feature_columns=feature_columns,
        target_column=args.target_column,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
    )
    test_dataset, test_timestamps = build_dataset(
        table=test_table,
        feature_columns=feature_columns,
        target_column=args.target_column,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = RecurrentForecaster(
        input_size=len(feature_columns),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        cell_type=args.model,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_state = None
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    test_loss, test_predictions, test_targets = evaluate(model, test_loader, criterion, device)

    test_predictions = target_scaler.inverse_transform(test_predictions).reshape(-1)
    test_targets = target_scaler.inverse_transform(test_targets).reshape(-1)
    metrics = regression_metrics(test_targets, test_predictions)

    model_path = args.output_dir / f"{args.model}_weather_forecast_model.pt"
    torch.save(model.state_dict(), model_path)
    save_loss_plot(train_losses, val_losses, args.output_dir / "loss_curve.png")
    save_prediction_plot(
        y_true=test_targets,
        y_pred=test_predictions,
        output_path=args.output_dir / "forecast_plot.png",
        timestamps=test_timestamps,
    )
    predictions_csv_path = args.output_dir / "predictions.csv"
    save_predictions_csv(
        timestamps=test_timestamps,
        y_true=test_targets,
        y_pred=test_predictions,
        output_path=predictions_csv_path,
    )
    save_json(
        {
            "model": args.model,
            "device": str(device),
            "test_loss": test_loss,
            "metrics": metrics,
            "data_path": str(args.data_path),
            "target_column": args.target_column,
            "feature_columns": feature_columns,
            "sequence_length": args.sequence_length,
            "forecast_horizon": args.forecast_horizon,
            "row_counts": {
                "full": len(table),
                "train": len(train_table),
                "validation": len(val_table),
                "test": len(test_table),
            },
            "config": vars(args),
        },
        args.output_dir / "metrics.json",
    )

    print("\nDataset summary")
    print(f"Rows            : {len(table)}")
    print(f"Target          : {args.target_column}")
    print(f"Features        : {', '.join(feature_columns)}")
    print(f"Saved model     : {model_path.resolve()}")
    print(f"Predictions CSV : {predictions_csv_path.resolve()}")
    print("\nFinal test metrics")
    print(f"MAE             : {metrics['mae']:.4f}")
    print(f"RMSE            : {metrics['rmse']:.4f}")
    print(f"Artifacts saved : {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
