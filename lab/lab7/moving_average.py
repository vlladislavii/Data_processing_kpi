import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(true, pred):
    """
    Обчислення метрик точності:
    MSE  – середньоквадратична помилка
    RMSE – корінь середньоквадратичної помилки
    MAE  – середня абсолютна помилка
    MAPE – середня абсолютна відносна помилка
    """
    true_vals = np.array(true)
    pred_vals = np.array(pred)
    # MSE та RMSE
    mse = mean_squared_error(true_vals, pred_vals)
    rmse = np.sqrt(mse)
    # MAE
    mae = mean_absolute_error(true_vals, pred_vals)
    # MAPE (ігноруємо нульові значення)
    mask = true_vals != 0
    mape = np.mean(np.abs((true_vals[mask] - pred_vals[mask]) / true_vals[mask])) * 100
    return mse, rmse, mae, mape

# --- Завантаження даних із файлу ---
# Формат дати: день/місяць/рік
data = pd.read_csv(
    'data.csv',
    index_col='Date',
    parse_dates=True,
    dayfirst=True
)
# Сортуємо за датами
data = data.sort_index()
# Визначаємо частоту або припускаємо щоденну
freq = pd.infer_freq(data.index) or 'D'
series = data['Value']

# --- Розбиття даних на тренувальні та тестові ---
# Останні 7 спостережень — тестова вибірка
train, test = series[:-7], series[-7:]

# --- Прогноз ковзним середнім ---
# Функція прогнозу з ковзним середнім по останніх window значеннях
def moving_average_forecast(history, window, horizon):
    hist = list(history)
    preds = []
    for _ in range(horizon):
        avg = np.mean(hist[-window:])
        preds.append(avg)
        hist.append(avg)
    return np.array(preds)

window = 5  # розмір вікна для ковзного середнього
horizon = 7  # кількість кроків прогнозу
ma_vals = moving_average_forecast(train.values, window, horizon)

# Індекс дат для прогнозу
t0 = train.index[-1]
forecast_index = pd.date_range(start=t0 + pd.Timedelta(days=1), periods=horizon, freq=freq)
ma_forecast = pd.Series(ma_vals, index=forecast_index)

# --- Обчислення метрик ---
mse, rmse, mae, mape = calculate_metrics(test.values, ma_vals)
print(f"MA (window={window}) → MSE: {mse:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape:.2f}%")

# --- Візуалізація результатів ---
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Навчальна вибірка', color='blue')
plt.plot(test.index, test, label='Тестова вибірка', color='black')
plt.plot(ma_forecast.index, ma_forecast, label='Прогноз MA', color='green', linestyle='--')
plt.title(f'Прогноз ковзним середнім (window={window}) vs Фактичні дані')
plt.xlabel('Дата')
plt.ylabel('Значення')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
