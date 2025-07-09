CONFIG = {
    'random_seed': 42,
    'train_test_split': 0.8,
    'scaler': 'minmax',
    # ARIMA
    'arima_order': (7, 1, 4),
    # LSTM
    'lstm_time_steps': 10,
    'lstm_units': 50,
    'lstm_dense_units': 25,
    'lstm_dropout': 0.2,
    'lstm_epochs': 50,
    'lstm_batch_size': 16,
    # Data
    'csv_path': 'data/air-quality-india.csv',
    'target_column': 'Average_PM2.5',
    'date_column': 'Date',
} 