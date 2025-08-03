# exp_smoothing_forecast_ua.py
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Приховуємо попередження щодо частоти
warnings.filterwarnings(
    'ignore',
    message='No frequency information was provided, so inferred frequency.*',
    category=UserWarning
)

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
    mse = mean_squared_error(true_vals, pred_vals)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_vals, pred_vals)
    # Уникаємо ділення на нуль
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
train, test = series[:-7], series[-7:]

# --- Побудова моделі експоненційного згладжування ---
# Simple Exponential Smoothing без тренду та сезонності
alpha = 0.3  # коефіцієнт згладжування (0 < α < 1)
model = ExponentialSmoothing(train, trend=None, seasonal=None)
fit = model.fit(smoothing_level=alpha, optimized=False)

# --- Прогнозування на 7 днів уперед ---
forecast = fit.forecast(7)

# --- Обчислення метрик ---
mse, rmse, mae, mape = calculate_metrics(test.values, forecast.values)
print(f"Exp. Smoothing (α={alpha}) → MSE: {mse:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape:.2f}%")

# --- Візуалізація результатів ---
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Навчальна вибірка', color='blue')
plt.plot(test.index, test, label='Тестова вибірка', color='black')
plt.plot(forecast.index, forecast, label='Прогноз ES', color='orange', linestyle='--')
plt.title(f'Експоненційне згладжування (α={alpha}) vs Фактичні дані')
plt.xlabel('Дата')
plt.ylabel('Значення')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()