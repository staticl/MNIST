"""
This script runs the test set on the trained model and shows the true and predicted values for every image in the set.

A pretrained model must be present.

Run this script via: `python -m main`
"""


import numpy as np
import matplotlib.pyplot as plt

import os
from pathlib import Path

from PIL import Image

from utils import math_fn

from src.data.datamanager import DataManager

from src.nn.model import MnistCNN


PROCESSED_PATH = os.path.join(Path(r"data"), Path(r"mnist_preprocessed.npz")) # path to the preprocessed data

SEED = 101 # seed used for reproducibility

MODEL = Path(r"mnistCNN_model.npz") # path to pretrained model

CONV_LAYER_PARAMS = [(16, 1, 3), (32, 16, 3), (64, 32, 3)] # parameters for every convolution layer (num_filters, in_channels, kernel_width), in_channels[i] = num_filters[i - 1]

def show_prediction(X: np.ndarray, true: np.ndarray, pred: np.ndarray, n: int = 4, m: int = 8) -> None:
    fig, axis = plt.subplots(n, m, figsize=(11.69, 8.27))
    if m * n != len(pred):
        raise ValueError("The size of the figure has the be the same as the number of elements in a batch.")
    for i in range(n):
        for j in range(m):
            transformed_X = (X[i * m + j] + 1) * 127.5
            image = Image.fromarray(transformed_X.squeeze())
            axis[i][j].imshow(image, cmap="gray")
            color = 'blue' if true[i * m + j] == pred[i * m + j] else 'red'
            axis[i][j].set_title(f"True: {true[i * m + j]}, Pred: {pred[i * m + j]}", fontsize=8, color=color)

            axis[i][j].axis("off")

    plt.tight_layout()
    plt.show()

def show_accuracy(accuracies: list[float]) -> None:
    fig, axis = plt.subplots(figsize=(8,6))
    fig.suptitle(f"Total Test Accuracy, Mean: {np.mean(accuracies)}")
    axis.plot(accuracies, marker="+", color="orange", label="accuracy per batch")
    axis.set_xlabel("epoch")
    axis.set_ylabel("accuracy")
    axis.grid()
    axis.legend()

    plt.tight_layout()
    plt.show()

def main() -> None:
    seed = SEED

    if not os.path.exists(MODEL):
        raise ValueError(f"A trained model hasn't been created yet.")

    data_manager = DataManager(PROCESSED_PATH, seed=seed)
    _, _, test_loader = data_manager.get_loader()
    n_batches = len(test_loader)

    print(f"Total number of batches: {n_batches}")

    seed = None if seed is None else seed + 1

    model = MnistCNN(CONV_LAYER_PARAMS, seed=seed)

    model_parameters = np.load(MODEL, allow_pickle=True)
    
    saved_params = list(model_parameters["parameters"])
    for saved_param, (params, _) in zip(saved_params, model.parameters()):
        params[:] = saved_param

    accuracies = []
    for X, y in test_loader:
        logits = model(X)
        pred = np.argmax(math_fn.softmax(logits), axis=1)
        show_prediction(X, y, pred) # plots the MNIST images with its true and predicted values
        accuracy = np.mean(y == pred)
        accuracies.append(accuracy)

    show_accuracy(accuracies)

    
if __name__ == "__main__":
    main()