# Weather forecasting benchmark

This run used the checked-in Open-Meteo snapshot for Ho Chi Minh City from 2025-01-01 through 2025-03-31. The CSV contains 2,160 complete hourly records and uses local time (Asia/Ho_Chi_Minh). Data by [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Reproduce

Run from the repository root with Python 3.11, PyTorch 2.5.1+cpu and NumPy 1.26.4. The reported run on Windows used two OpenMP and MKL threads.

```bash
python train.py --data-path docs/media/weather_data.csv --model rnn --epochs 10 --device cpu --output-dir artifacts/benchmark-rnn
python train.py --data-path docs/media/weather_data.csv --model lstm --epochs 10 --device cpu --output-dir artifacts/benchmark-lstm
python train.py --data-path docs/media/weather_data.csv --model gru --epochs 10 --device cpu --output-dir artifacts/benchmark-gru
```

The first 70% of rows are used for training and the next 15% for validation. Validation MSE selects the checkpoint within each run. The remaining 324 records yield 300 test windows. For every window the persistence forecast uses the last observed temperature in the same input, and both MAE and RMSE use degrees Celsius.

- [benchmark.json](benchmark.json) summarizes all three runs and persistence.
- [metrics.json](metrics.json), [forecast plot](forecast_plot.png), [loss curve](loss_curve.png), and [predictions.csv](predictions.csv) are from the RNN run.
- [weather_data.csv](weather_data.csv) is the snapshot used for all runs.

The CSV contains no missing values, so interpolation does not affect these scores. Results are from one seed, one period and one city.
