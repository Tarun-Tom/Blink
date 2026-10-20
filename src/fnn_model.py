import tensorflow as tf
from tensorflow.keras import layers, models

def build_fnn(
    input_shape: tuple = (64, 64, 1),
    learning_rate: float = 0.001,
    dropout_rate: float = 0.3
) -> tf.keras.Model:
    """
    Builds the baseline Feedforward Neural Network (FNN) for eye-state classification.
    
    Architecture:
    - Input: (64, 64, 1) grayscale
    - Rescaling: 1.0 / 255.0 to normalize pixel values into [0, 1]
    - Flatten: converts 64x64x1 (4096 values) into a 1D feature vector
    - Dense(512, relu) + Dropout(0.3)
    - Dense(256, relu) + Dropout(0.3)
    - Dense(128, relu) + Dropout(0.3)
    - Dense(1, sigmoid): binary probability output (0 = sleepy/closed, 1 = awake/open)
    """
    model = models.Sequential([
        layers.Input(shape=input_shape, name="input_image"),
        layers.Rescaling(1.0 / 255.0, name="rescaling_norm"),
        layers.Flatten(name="flatten"),
        layers.Dense(512, activation="relu", name="dense_512"),
        layers.Dropout(dropout_rate, name="dropout_1"),
        layers.Dense(256, activation="relu", name="dense_256"),
        layers.Dropout(dropout_rate, name="dropout_2"),
        layers.Dense(128, activation="relu", name="dense_128"),
        layers.Dropout(dropout_rate, name="dropout_3"),
        layers.Dense(1, activation="sigmoid", name="output_sigmoid")
    ], name="FNN_Baseline")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
