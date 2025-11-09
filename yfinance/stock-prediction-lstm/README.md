# Stock Price Prediction using LSTM

This project implements Long Short-Term Memory (LSTM) models to predict stock prices based on historical data. The models are trained using data fetched from the yfinance library.

## Project Structure

```
stock-prediction-lstm
├── data
│   └── README.md
├── models
│   └── README.md
├── notebooks
│   └── exploratory_analysis.ipynb
├── src
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd stock-prediction-lstm
pip install -r requirements.txt
```

## Usage

1. **Data Loading**: Use the `DataLoader` class in `src/data_loader.py` to fetch and preprocess historical stock data.
2. **Model Training**: Define the LSTM model architecture using the `LSTMModel` class in `src/model.py`, and train it using the `train_model` function in `src/train.py`.
3. **Evaluation**: After training, evaluate the model's performance using the utility functions in `src/utils.py`.
4. **Exploratory Analysis**: Explore the dataset and visualize insights using the Jupyter notebook located in the `notebooks` directory.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.