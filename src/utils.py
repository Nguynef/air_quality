import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

def plot_time_series(df, date_col, value_col, label=None, title=None, xlabel=None, ylabel=None):
    plt.figure(figsize=(12, 6))
    plt.plot(df[date_col], df[value_col], linestyle='-', color='b', label=label or value_col)
    plt.xlabel(xlabel or date_col)
    plt.ylabel(ylabel or value_col)
    plt.title(title or f"{value_col} theo thời gian")
    plt.grid(True)
    if label:
        plt.legend()
    plt.show()

def plot_box(df, group_col, value_col, title=None, xlabel=None, ylabel=None, rotation=0):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=group_col, y=value_col, data=df)
    plt.title(title or f"{value_col} by {group_col}")
    plt.xlabel(xlabel or group_col)
    plt.ylabel(ylabel or value_col)
    plt.xticks(rotation=rotation)
    plt.grid(True)
    plt.show()

def plot_seasonal_boxes(df, date_col, value_col):
    df = df.copy()
    df['Month'] = df[date_col].dt.month
    df['Year'] = df[date_col].dt.year
    plot_box(df, 'Month', value_col, title=f"Xu hướng theo tháng của {value_col}", xlabel="Month", ylabel=value_col)
    plot_box(df, 'Year', value_col, title=f"Xu hướng theo năm của{value_col}", xlabel="Year", ylabel=value_col, rotation=45)

def plot_decomposition(df, date_col, value_col, period=30, model='additive'):
    df = df.copy()
    df = df.set_index(date_col)
    df = df.asfreq("D")
    result = seasonal_decompose(df[value_col].interpolate(), model=model, period=period)
    plt.figure(figsize=(12, 10))
    plt.subplot(411)
    plt.plot(result.observed, color="blue", linewidth=1.5)
    plt.title('Observed', fontsize=12, fontweight="bold")
    plt.subplot(412)
    plt.plot(result.trend, color="green", linewidth=1.5)
    plt.title('Trend', fontsize=12, fontweight="bold")
    plt.subplot(413)
    plt.plot(result.seasonal, color="orange", linewidth=1.5)
    plt.title('Seasonality', fontsize=12, fontweight="bold")
    plt.subplot(414)
    plt.plot(result.resid, color="red", linewidth=1.5)
    plt.axhline(0, linestyle="--", color="black", linewidth=1)
    plt.title('Residuals (Noise)', fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

def check_stationarity(df, column, alpha=0.05, print_result=True):
    result = adfuller(df[column])
    adf_stat, p_value, _, _, critical_values, _ = result
    if print_result:
        print(f"ADF Statistic: {adf_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print("Critical Values:")
        for key, value in critical_values.items():
            print(f"   {key}: {value:.4f}")
        if p_value < alpha:
            print("Chuỗi có tính dừng.")
        else:
            print("Chuỗi vẫn chưa dừng, có thể cần lấy sai phân bậc cao hơn.")
    return {'adf_stat': adf_stat, 'p_value': p_value, 'critical_values': critical_values}

def compare_predicted_actual(df, date_col, predicted_col, actual_col, title=None, xlabel=None, ylabel=None):
    plt.figure(figsize=(12, 6))
    plt.plot(df[date_col], df[predicted_col], label="Predicted", color="blue")
    plt.plot(df[date_col], df[actual_col], label="Actual", color="red")
    plt.xlabel(xlabel or date_col)
    plt.ylabel(ylabel or predicted_col)
    plt.title(title or f"So sánh dự đoán và thực tế {predicted_col}")
    plt.grid(True)
    plt.legend()
    plt.show()
