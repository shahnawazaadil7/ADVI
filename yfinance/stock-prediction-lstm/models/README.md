# LSTM Stock Price Prediction Model

This directory contains the implementation of Long Short-Term Memory (LSTM) models for predicting stock prices based on historical data.

## Model Architecture

The LSTM model is designed to capture the temporal dependencies in stock price data. It consists of the following layers:

- **Input Layer**: Accepts the preprocessed stock price data.
- **LSTM Layers**: One or more LSTM layers that learn the temporal patterns in the data.
- **Dense Layer**: A fully connected layer that outputs the predicted stock price.

## Training Process

The model is trained using historical stock price data fetched from the yfinance library. The training process involves:

1. **Data Loading**: Historical stock data is loaded and preprocessed.
2. **Model Compilation**: The model is compiled with a specified loss function and optimizer.
3. **Model Fitting**: The model is trained on the training dataset for a specified number of epochs.

## Evaluation Metrics

The performance of the LSTM model is evaluated using the following metrics:

- **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in a set of predictions.
- **Root Mean Squared Error (RMSE)**: Measures the square root of the average of squared differences between predicted and actual values.

## Usage

To use the LSTM model for stock price prediction, follow these steps:

1. Load the historical stock data using the `DataLoader` class.
2. Preprocess the data to make it suitable for training.
3. Create an instance of the `LSTMModel` class and build the model.
4. Train the model using the `train_model` function.
5. Evaluate the model performance using the provided metrics.

For detailed instructions, refer to the main README file in the project root.