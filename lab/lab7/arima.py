import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
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
    RMSPE – корінь середньої квадратичної відносної помилки
    """
    true_vals = np.array(true)
    pred_vals = np.array(pred)
    mse = mean_squared_error(true_vals, pred_vals)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_vals, pred_vals)
    mask = true_vals != 0
    mape = np.mean(np.abs((true_vals[mask] - pred_vals[mask]) / true_vals[mask])) * 100
    rmspe = np.sqrt(np.mean(((true_vals[mask] - pred_vals[mask]) / true_vals[mask])**2)) * 100
    return mse, rmse, mae, mape, rmspe

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
# Використовуємо останні 7 спостережень для тестування
train, test = series[:-7], series[-7:]

# --- Побудова ARIMA(p,d,q) моделі ---
# Налаштуйте p, d, q за потреби
p, d, q = 2, 1, 2
model = ARIMA(
    train,
    order=(p, d, q),
    enforce_stationarity=False,
    enforce_invertibility=False
)
fit = model.fit()

# --- Прогнозування на 7 днів уперед ---
forecast = fit.get_forecast(steps=7)
# Індекс для прогнозу
t0 = train.index[-1]
forecast_index = pd.date_range(start=t0 + pd.Timedelta(days=1), periods=7, freq=freq)
forecast_values = pd.Series(forecast.predicted_mean.values, index=forecast_index)

# --- Обчислення метрик ---
mse, rmse, mae, mape, rmspe = calculate_metrics(test.values, forecast_values.values)
print(f"ARIMA({p},{d},{q}) → MSE: {mse:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape:.2f}%, RMSPE: {rmspe:.2f}%")

# --- Візуалізація результатів ---
plt.figure(figsize=(10, 5))
plt.plot(train.index, train, label='Навчальна вибірка', color='blue')
plt.plot(test.index, test, label='Тестова вибірка', color='black')
plt.plot(forecast_values.index, forecast_values, label='Прогноз ARIMA', color='blue', linestyle='--')
plt.title(f'ARIMA({p},{d},{q}) Прогноз vs Фактичні дані')
plt.xlabel('Дата')
plt.ylabel('Значення')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()