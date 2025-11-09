# This file contains information about the dataset used for training the LSTM models, including its source and structure.

# Dataset Information

The dataset used for training the LSTM models consists of historical stock price data. This data is sourced from Yahoo Finance using the `yfinance` library.

## Data Source

- **Provider**: Yahoo Finance
- **Library Used**: yfinance

## Data Structure

The dataset typically includes the following columns:

- **Date**: The date of the stock price entry.
- **Open**: The price at which the stock opened on that date.
- **High**: The highest price of the stock during the trading session.
- **Low**: The lowest price of the stock during the trading session.
- **Close**: The price at which the stock closed on that date.
- **Volume**: The number of shares traded during the session.
- **Adj Close**: The adjusted closing price, accounting for dividends and stock splits.

## Usage

The data will be loaded and preprocessed using the `DataLoader` class defined in `src/data_loader.py`. The preprocessed data will then be used to train the LSTM models for stock price prediction.