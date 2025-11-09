def plot_results(actual, predicted):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 7))
    plt.plot(actual, color='blue', label='Actual Stock Price')
    plt.plot(predicted, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def calculate_metrics(actual, predicted):
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    return {'Mean Squared Error': mse, 'Mean Absolute Error': mae}