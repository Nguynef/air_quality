import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from src.preprocessing import preprocessing
from src.utils import (
    plot_time_series,
    plot_seasonal_boxes,
    plot_decomposition,
    check_stationarity,
    compare_predicted_actual
)
from src.models import build_arima_model, build_lstm_model, train_lstm_model, complex_lstm_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs('plots', exist_ok=True)

data = pd.read_csv("data/air-quality-india.csv")
df_daily, scaler = preprocessing(data, scale=True)
df_daily["Scaled_PM2.5"] = scaler.transform(df_daily[["Average_PM2.5"]])

check_stationarity(df_daily, "Average_PM2.5")

plot_time_series(
    df_daily,
    date_col="Date",
    value_col="Average_PM2.5",
    label="PM2.5 Levels",
    title="Mức PM2.5 theo thời gian",
    xlabel="Date",
    ylabel="Average PM2.5 Level",
    save_path="plots/time_series.png"
)
plot_seasonal_boxes(
    df_daily,
    date_col="Date",
    value_col="Average_PM2.5",
    save_dir="plots"
)
plot_decomposition(
    df_daily,
    date_col="Date",
    value_col="Average_PM2.5",
    period=30,
    save_path="plots/decomposition.png"
)

df_daily["PM2.5_diff"] = df_daily["Average_PM2.5"].diff()
df_daily.dropna(inplace=True)

check_stationarity(df_daily, "PM2.5_diff")

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(df_daily["PM2.5_diff"].dropna(), lags=30, ax=axes[0])
axes[0].set_title("ACF - Xác định q")
plot_pacf(df_daily["PM2.5_diff"].dropna(), lags=30, ax=axes[1])
axes[1].set_title("PACF - Xác định p")
plt.tight_layout()
plt.savefig("plots/acf_pacf.png", bbox_inches='tight')
plt.close()

arima_order = (7, 1, 4)
model_fit_arima = build_arima_model(df_daily["Average_PM2.5"], order=arima_order)
print(model_fit_arima.summary())

df_daily["Predicted_PM2.5"] = model_fit_arima.fittedvalues
compare_predicted_actual(
    df_daily,
    date_col="Date",
    predicted_col="Predicted_PM2.5",
    actual_col="Average_PM2.5",
    title="So sánh giá trị thực tế và dự đoán bằng ARIMA",
    xlabel="Ngày",
    ylabel="PM2.5",
    save_path="plots/arima_pred_vs_actual.png"
)

def print_metrics(y_true, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2:  {r2:.4f}")

print_metrics(df_daily["Average_PM2.5"].iloc[-len(df_daily["Predicted_PM2.5"]):], df_daily["Predicted_PM2.5"], model_name="ARIMA")

train_size = int(len(df_daily) * 0.8)
train, test = df_daily.iloc[:train_size], df_daily.iloc[train_size:]

def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 10
X_train, y_train = create_sequences(train["Scaled_PM2.5"].values, time_steps)
X_test, y_test = create_sequences(test["Scaled_PM2.5"].values, time_steps)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print("Shape của X_train:", X_train.shape)
print("Shape của y_train:", y_train.shape)

lstm_model = build_lstm_model(input_shape=(time_steps, 1), lstm_units=50, dense_units=25, dropout_rate=0.2)
history = train_lstm_model(lstm_model, X_train, y_train, X_test, y_test, epochs=50, batch_size=16, verbose=1)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("plots/lstm_loss.png", bbox_inches='tight')
plt.close()

lstm_pred_scaled = lstm_model.predict(X_test)
y_pred_inv = scaler.inverse_transform(lstm_pred_scaled)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
lstm_compare_df = pd.DataFrame({
    "Date": df_daily["Date"].iloc[-len(y_test):].values,
    "Actual": y_test_inv.flatten(),
    "Predicted": y_pred_inv.flatten()
})
compare_predicted_actual(
    lstm_compare_df,
    date_col="Date",
    predicted_col="Predicted",
    actual_col="Actual",
    title="Dự báo PM2.5 sử dụng LSTM",
    xlabel="Ngày",
    ylabel="PM2.5",
    save_path="plots/lstm_pred_vs_actual.png"
)
print_metrics(lstm_compare_df["Actual"], lstm_compare_df["Predicted"], model_name="LSTM")

# --- Complex LSTM Model ---
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(patience=5, factor=0.5)
]
complex_lstm = complex_lstm_model(input_shape=(time_steps, 1), lstm_units=64, dense_units=32, dropout_rate=0.2)
complex_history = train_lstm_model(complex_lstm, X_train, y_train, X_test, y_test, epochs=50, batch_size=16, verbose=1, callbacks=callbacks)

plt.figure(figsize=(8, 5))
plt.plot(complex_history.history['loss'], label='Training Loss')
plt.plot(complex_history.history['val_loss'], label='Validation Loss')
plt.title('Complex LSTM Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("plots/lstm_complex_loss.png", bbox_inches='tight')
plt.close()

complex_lstm_pred_scaled = complex_lstm.predict(X_test)
complex_y_pred_inv = scaler.inverse_transform(complex_lstm_pred_scaled)
complex_y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
complex_lstm_compare_df = pd.DataFrame({
    "Date": df_daily["Date"].iloc[-len(y_test):].values,
    "Actual": complex_y_test_inv.flatten(),
    "Predicted": complex_y_pred_inv.flatten()
})
compare_predicted_actual(
    complex_lstm_compare_df,
    date_col="Date",
    predicted_col="Predicted",
    actual_col="Actual",
    title="Dự báo PM2.5 sử dụng Complex LSTM",
    xlabel="Ngày",
    ylabel="PM2.5",
    save_path="plots/lstm_complex_pred_vs_actual.png"
)
print_metrics(complex_lstm_compare_df["Actual"], complex_lstm_compare_df["Predicted"], model_name="Complex LSTM")

