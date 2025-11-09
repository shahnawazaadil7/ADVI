def train_model(model, train_data, epochs=50, batch_size=32, save_path='model.h5'):
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.models import load_model

    # Unpack training data
    X_train, y_train = train_data

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                              validation_split=0.2, callbacks=[early_stopping])

    # Save the trained model
    model.model.save(save_path)

    return history