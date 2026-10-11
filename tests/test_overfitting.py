"""
This is a test script used to validate the implementation of the CNN by testing for overfitting.

Run this script via: `python -m tests.test_overfitting` 
"""


import numpy as np

import os
from pathlib import Path

from src.train.train import evalute, cnn_init
from src.train.plot import plot_training_loss


PROCESSED_PATH = os.path.join(Path(r"data"), Path(r"mnist_preprocessed.npz"))

SEED = 101

EPOCHS = 1000

CONV_LAYER_PARAMS = [(16, 1, 6), (32, 16, 8), (64, 32, 10)]


def test_overfitting():
    cnn_componentes = cnn_init(PROCESSED_PATH, CONV_LAYER_PARAMS, seed=SEED)

    train_loader = cnn_componentes["train_loader"]

    model = cnn_componentes["model"]
    crossEntropyWithLogitsLoss = cnn_componentes["crossEntropyWithLogitsLoss"]

    optimizer = cnn_componentes["optimizer"]

    X_overfit, y_overfit = None, None

    for X, y in train_loader:
        X_overfit = X[0][np.newaxis]
        y_overfit = np.array([y[0]])
        break

    losses = []
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        logits = model(X_overfit)
        loss = crossEntropyWithLogitsLoss(logits, y_overfit)
        losses.append(loss)
        dout = crossEntropyWithLogitsLoss.backward()
        model.backward(dout)
        optimizer.step()

    plot_training_loss(losses)


if __name__ == "__main__":
    test_overfitting()
