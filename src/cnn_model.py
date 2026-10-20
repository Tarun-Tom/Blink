import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(
    input_shape: tuple = (64, 64, 1),
    learning_rate: float = 0.001,
    dropout_rate: float = 0.3
) -> tf.keras.Model:
    """
    Builds a compact Convolutional Neural Network (CNN) for eye-state classification.
    
    Architecture:
    - Input: (64, 64, 1) grayscale
    - Rescaling: 1.0 / 255.0 to normalize pixel values into [0, 1]
    - Block 1: Conv2D(32, 3x3, relu, padding='same') + MaxPooling2D(2x2) -> (32, 32, 32)
    - Block 2: Conv2D(64, 3x3, relu, padding='same') + MaxPooling2D(2x2) -> (16, 16, 64)
    - Block 3: Conv2D(128, 3x3, relu, padding='same') + MaxPooling2D(2x2) -> (8, 8, 128)
    - Classifier: Flatten -> Dense(128, relu) -> Dropout(0.3) -> Dense(1, sigmoid)
    """
    model = models.Sequential([
        layers.Input(shape=input_shape, name="input_image"),
        layers.Rescaling(1.0 / 255.0, name="rescaling_norm"),
        
        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1"),
        layers.MaxPooling2D((2, 2), name="pool1"),
        
        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2"),
        layers.MaxPooling2D((2, 2), name="pool2"),
        
        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3"),
        layers.MaxPooling2D((2, 2), name="pool3"),
        
        # Dense Classifier
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense_128"),
        layers.Dropout(dropout_rate, name="dropout"),
        layers.Dense(1, activation="sigmoid", name="output_sigmoid")
    ], name="CNN_Baseline")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
