# Kết quả chạy minh họa dự báo nhiệt độ

Các file trong thư mục này được tạo bằng mã nguồn gốc tại commit `09026ec8cc86c0088d3838ee10cde24395e0cf7c`, ngày 05/11/2026. Không thay đổi thuật toán huấn luyện để tạo minh họa.

## Nguồn và tái tạo

Dữ liệu do [Open-Meteo](https://open-meteo.com/) cung cấp qua [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), giấy phép [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Dữ liệu được lấy theo tọa độ TP.HCM; đây là dữ liệu lịch sử/tái phân tích, không phải số đo từ một trạm do dự án vận hành. Biểu đồ và cột dự đoán là kết quả xử lý bằng mô hình của kho này.

Từ thư mục gốc dự án, trong môi trường Python đã cài thư viện:

```bash
python scripts/fetch_weather_data.py --latitude 10.8231 --longitude 106.6297 --start-date 2025-01-01 --end-date 2025-03-31 --timezone Asia/Ho_Chi_Minh
python train.py --data-path data/weather_data.csv --model lstm --epochs 10 --device cpu --output-dir artifacts/demo
```

Môi trường: Python 3.11.15, torch 2.5.1+cpu, NumPy 1.26.4, Matplotlib 3.11.2, requests 2.34.2; CPU Windows. Lần chạy giới hạn `OMP_NUM_THREADS=2` và `MKL_NUM_THREADS=2`.

Sao chép `forecast_plot.png`, `loss_curve.png`, `metrics.json`, `predictions.csv` từ `artifacts/demo/` sang thư mục này. `weather_data.csv` lưu bản dữ liệu đầu vào dùng cho minh họa để có thể chạy lại mà không phụ thuộc thay đổi của API. Không đưa checkpoint vào Git.

SHA-256 của `weather_data.csv`: `6257497c3e60874cdfaaf18ee4033f6701ccc2d0c0fb2cd33c87466d8ed610e2`.

Để dùng bản dữ liệu này, thay đường dẫn trong lệnh train bằng `--data-path docs/media/weather_data.csv`.

## Diễn giải

- 2.160 dòng dữ liệu; chia 1.512 / 324 / 324 theo thời gian.
- 1.488 cửa sổ train, 300 validation, 300 test; mỗi cửa sổ 24 bước, dự đoán 1 bước tiếp theo.
- MAE = 0,9233878851 °C; RMSE = 1,1695268154 °C.
- MSE test trong JSON dùng mục tiêu đã chuẩn hóa, không phải °C² trực tiếp.
- `predictions.csv` chứa các dự đoán trên lịch sử được giữ lại, không phải chuỗi dự báo tương lai ngoài dữ liệu.
- Kết quả phản ánh mã hiện tại, gồm giới hạn lưu best checkpoint trên CPU đã ghi trong README gốc. Không dùng số liệu này làm khẳng định vượt trội so với phương pháp khác.
