from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

def build_lstm_model(input_shape, num_classes):
    """
    Builds an LSTM model for multiclass classification.

    Args:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of target classes.

    Returns:
        model: Compiled Keras model.
    """
    model = Sequential([
        # LSTM layer for sequence modeling
        LSTM(128, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer='l2'),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_lstm_model(X_train, y_train, input_shape, num_classes):
    """
    Trains the LSTM model.

    Args:
        X_train: Training features.
        y_train: Training labels.
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of target classes.

    Returns:
        model: Trained Keras model.
    """
    model = build_lstm_model(input_shape, num_classes)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    history = model.fit(
        X_train, y_train, validation_split=0.2, epochs=30, batch_size=32, callbacks=[lr_scheduler]
    )
    return model, history
