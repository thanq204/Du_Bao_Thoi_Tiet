from __future__ import annotations

import argparse
import csv
from pathlib import Path

import requests


DEFAULT_HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical weather data from Open-Meteo and save it as CSV.")
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--start-date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--timezone", type=str, default="auto")
    parser.add_argument("--hourly", type=str, default=",".join(DEFAULT_HOURLY_VARIABLES))
    parser.add_argument("--output-path", type=Path, default=Path("data/weather_data.csv"))
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hourly_variables = [item.strip() for item in args.hourly.split(",") if item.strip()]

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": args.latitude,
        "longitude": args.longitude,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "hourly": ",".join(hourly_variables),
        "timezone": args.timezone,
    }

    response = requests.get(url, params=params, timeout=args.timeout)
    response.raise_for_status()
    payload = response.json()

    hourly_payload = payload.get("hourly")
    if not hourly_payload or "time" not in hourly_payload:
        raise RuntimeError("Open-Meteo response does not contain the expected hourly data.")

    time_values = hourly_payload["time"]
    csv_columns = ["timestamp"] + hourly_variables
    rows = []
    for index, timestamp in enumerate(time_values):
        row = {"timestamp": timestamp}
        for column in hourly_variables:
            series = hourly_payload.get(column)
            row[column] = "" if series is None or index >= len(series) else series[index]
        rows.append(row)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {args.output_path.resolve()}")
    print(f"Columns: {', '.join(csv_columns)}")


if __name__ == "__main__":
    main()
