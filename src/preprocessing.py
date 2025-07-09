import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler

def preprocessing(df, scale=False):
    df = df.dropna(subset=["Timestamp", "PM2.5"]).copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df[["Timestamp", "PM2.5"]]
    df_daily = df.groupby(df["Timestamp"].dt.date)["PM2.5"].mean().reset_index()
    df_daily.rename(columns={"Timestamp": "Date", "PM2.5": "Average_PM2.5"}, inplace=True)
    df_daily["Date"] = pd.to_datetime(df_daily["Date"])
    if scale:
        scaler = MinMaxScaler()
        df_daily["Average_PM2.5"] = scaler.fit_transform(df_daily[["Average_PM2.5"]])
        return df_daily, scaler
    return df_daily