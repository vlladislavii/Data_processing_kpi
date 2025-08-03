import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Приховуємо Warning щодо частоти
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
# Формат дати день/місяць/рік
data = pd.read_csv(
    'data.csv',
    index_col='Date',
    parse_dates=True,
    dayfirst=True
)
data = data.sort_index()
# Визначаємо частоту, але не встановлюємо її явно, щоб уникнути помилок
freq = pd.infer_freq(data.index)
# Якщо частота не виведена, припускаємо щоденну
freq = freq or 'D'
series = data['Value']

# --- Розбиття даних на тренувальні та тестові ---
train, test = series[:-12], series[-12:]

# --- Побудова ARMA(2,1) моделі через SARIMAX ---
model = SARIMAX(
    train,
    order=(2, 0, 1),
    seasonal_order=(0, 0, 0, 0),
    enforce_stationarity=False,
    enforce_invertibility=False
)
fit = model.fit(disp=False)

# --- Прогнозування на 12 днів уперед ---
pred = fit.forecast(steps=12)

# --- Обчислення метрик ---
mse, rmse, mae, mape, rmspe = calculate_metrics(test, pred)
print(f"ARMA(2,1) → MSE: {mse:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape:.2f}%, RMSPE: {rmspe:.2f}%")

# --- Візуалізація результатів ---
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Тренувальні дані', color='blue')
plt.plot(test.index, test, label='Тестові дані', color='black')
plt.plot(pred.index, pred, label='Прогноз ARMA', color='red', linestyle='--')
plt.title('ARMA(2,1) Прогноз vs Фактичні дані')
plt.xlabel('Дата')
plt.ylabel('Значення')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
