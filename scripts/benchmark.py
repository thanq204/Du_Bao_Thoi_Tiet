"""Compare RNN, LSTM and GRU with the same CPU configuration and dataset."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("rnn", "lstm", "gru")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=ROOT / "docs/media/weather_data.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/benchmark")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if not args.data_path.is_file():
        parser.error(f"data file not found: {args.data_path}")

    output_dir = args.output_dir.resolve()
    scores = {}
    for model in MODELS:
        model_output = output_dir / model
        subprocess.run(
            [
                sys.executable, str(ROOT / "train.py"),
                "--data-path", str(args.data_path.resolve()),
                "--model", model,
                "--epochs", str(args.epochs),
                "--device", "cpu",
                "--seed", str(args.seed),
                "--output-dir", str(model_output),
            ],
            check=True,
        )
        report = json.loads((model_output / "metrics.json").read_text(encoding="utf-8"))
        scores[model] = {
            **report["metrics"],
            "best_epoch": report["best_epoch"],
            "best_validation_loss": report["best_validation_loss"],
        }
        baseline = report["persistence_baseline"]
        if scores.setdefault("persistence", baseline) != baseline:
            raise RuntimeError("Persistence scores differ between runs on the same dataset.")

    summary = {
        "seed": args.seed,
        "epochs_per_model": args.epochs,
        "test_windows": report["row_counts"]["test"] - report["sequence_length"] - report["forecast_horizon"] + 1,
        "sequence_length": report["sequence_length"],
        "forecast_horizon": report["forecast_horizon"],
        "device": "cpu",
        "metrics_unit": "celsius",
        "scores": scores,
    }
    path = output_dir / "benchmark.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\nModel        MAE (°C)  RMSE (°C)")
    for model in (*MODELS, "persistence"):
        print(f"{model:<12} {scores[model]['mae']:>8.4f}  {scores[model]['rmse']:>9.4f}")
    print(f"Saved benchmark: {path}")


if __name__ == "__main__":
    main()
