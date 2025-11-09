class DataLoader:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def load_data(self):
        import yfinance as yf
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        return self.data

    def preprocess_data(self):
        import pandas as pd
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Keep only the 'Close' prices
        data = self.data[['Close']]
        # Normalize the data
        data['Close'] = (data['Close'] - data['Close'].min()) / (data['Close'].max() - data['Close'].min())
        
        # Create sequences for LSTM
        sequence_length = 60
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data.iloc[i:i + sequence_length].values)
            y.append(data.iloc[i + sequence_length].values)
        
        return np.array(X), np.array(y)