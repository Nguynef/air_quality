import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization


def build_arima_model(series, order):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit


def build_lstm_model(input_shape, lstm_units=50, dense_units=25, dropout_rate=0.2):
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units=lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=dense_units),
        Dense(units=1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def complex_lstm_model(input_shape, lstm_units=64, dense_units=32, dropout_rate=0.1):
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate),
        LSTM(units=lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=dense_units, activation='relu'),
        Dense(units=1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=16, verbose=1, callbacks=None):
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=verbose,
        callbacks=callbacks
    )
    return history 