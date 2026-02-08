"""Neural network architecture and training for tip prediction."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def build_model(input_dim: int, hidden_units: tuple[int, ...] = (128, 64),
                dropout_rate: float = 0.5,
                learning_rate: float = 0.001) -> Sequential:
    """Build a feed-forward regression network.

    Args:
        input_dim: Number of input features.
        hidden_units: Tuple of hidden layer sizes.
        dropout_rate: Dropout probability after the first hidden layer.
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential()

    for i, units in enumerate(hidden_units):
        kwargs = {"input_dim": input_dim} if i == 0 else {}
        model.add(Dense(units, activation="relu", **kwargs))
        if i == 0:
            model.add(Dropout(dropout_rate))

    # Linear output for regression
    model.add(Dense(1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
    )
    return model


def train_model(model: Sequential, X_train: np.ndarray, y_train,
                epochs: int = 50, batch_size: int = 8196,
                validation_split: float = 0.2,
                use_gpu: bool = True) -> tf.keras.callbacks.History:
    """Train the model with optional GPU acceleration.

    Args:
        model: Compiled Keras model.
        X_train: Scaled training features.
        y_train: Training target values.
        epochs: Number of training epochs.
        batch_size: Samples per gradient update.
        validation_split: Fraction of training data for validation.
        use_gpu: Whether to place training on GPU.

    Returns:
        Keras History object with training metrics.
    """
    device = "/GPU:0" if use_gpu and tf.config.list_physical_devices("GPU") else "/CPU:0"

    with tf.device(device):
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
        )
    return history
