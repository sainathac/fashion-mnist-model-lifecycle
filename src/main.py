import datetime
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def visualize_samples(images, labels):
    plt.figure(figsize=(8, 8))
    for index in range(16):
        plt.subplot(4, 4, index + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[index], cmap="gray")
        plt.xlabel(CLASS_NAMES[labels[index]])
    plt.tight_layout()
    plt.show()


def build_model(num_classes=10):
    model = Sequential(
        [
            Flatten(input_shape=(28, 28)),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    set_seeds()

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    print("Train shape:", x_train.shape, y_train.shape)
    print("Test shape:", x_test.shape, y_test.shape)

    visualize_samples(x_train, y_train)

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train_encoded = to_categorical(y_train, 10)
    y_test_encoded = to_categorical(y_test, 10)

    model = build_model()
    model.summary()

    log_dir = os.path.join(
        "logs",
        "fashion_mnist",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(
        x_train,
        y_train_encoded,
        epochs=15,
        batch_size=64,
        validation_split=0.1,
        callbacks=[tensorboard_callback],
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test_encoded, verbose=0)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "fashion_mnist_dense_model.keras")
    model.save(save_path)
    print("Model saved to:", save_path)

    loaded_model = load_model(save_path)
    print("\nLoaded model summary:")
    loaded_model.summary()

    original_params = model.count_params()
    loaded_params = loaded_model.count_params()

    print("\nOriginal model parameters:", original_params)
    print("Loaded model parameters:  ", loaded_params)

    if original_params == loaded_params:
        print("Parameter count is consistent between original and loaded model.")
    else:
        print("Parameter count mismatch between original and loaded model.")


if __name__ == "__main__":
    main()
