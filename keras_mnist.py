import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


def get_uncompiled_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(128, activation="relu", name="dense_1")(inputs)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_compiled_model(lr=0.001):
    model = get_uncompiled_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model


def get_mnist_data(data_dir, val_split_pct=0.1):

    # Get data_dir as pathlib.Path + make it absolute
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir).absolute()

    # If data_dir doesn't exist, we make that here
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    # Download and/or load data from disk
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(data_dir / 'mnist.npz')

    # Reformat
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Split train into train + val
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_split_pct)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def set_keras_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--val_split_pct', type=float, default=0.1)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='.tfds_data/')
    parser.add_argument('--logdir', type=str, default='./lightning_logs/keras/')
    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)
    set_keras_seed(args.seed)

    train_data, val_data, test_data = get_mnist_data(args.data_dir, args.val_split_pct)
    model = get_compiled_model(args.lr)

    print("Fit model on training data")
    history = model.fit(
        *train_data,
        batch_size=args.batch_size,
        epochs=args.max_epochs,
        validation_data=val_data,
        callbacks=[keras.callbacks.TensorBoard(log_dir=args.logdir)],
    )

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(*test_data, batch_size=128)
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    predictions = model.predict(x_test[:3])
    print("predictions shape:", predictions.shape)


if __name__ == '__main__':
    main()
