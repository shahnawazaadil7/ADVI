class LSTMModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=self.input_shape))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.num_classes))

    def compile_model(self, optimizer='adam', loss='mean_squared_error'):
        if self.model is None:
            raise Exception("Model is not built. Call build_model() before compiling.")
        self.model.compile(optimizer=optimizer, loss=loss)