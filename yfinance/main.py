import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def get_stock_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data[['Close']]

def prepare_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_predict_future(ticker, start, end, future_days=30, epochs=50, batch_size=32, time_step=60):
    data = get_stock_data(ticker, start, end)
    X, y, scaler = prepare_data(data.values, time_step)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    future_predictions = []
    last_sequence = X[-1]

    for _ in range(future_days):
        prediction = model.predict(last_sequence.reshape(1, time_step, 1))
        future_predictions.append(prediction[0, 0])

        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = prediction

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-100:], data['Close'].values[-100:], label='Actual Stock Price', color='blue')

    future_dates = pd.date_range(start=data.index[-1], periods=future_days+1)[1:]
    plt.plot(future_dates, future_predictions, label=f'Predicted Next {future_days} Days', color='red')

    plt.legend()
    plt.show()

train_and_predict_future('AAPL', '2015-01-01', '2023-12-31', future_days=30)