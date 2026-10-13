import numpy as np

import os
from pathlib import Path

from utils import math_fn

from src.data.datamanager import DataManager
from src.data.dataloader import Dataloader

from src.nn.model import MnistCNN
from src.nn.loss import CrossEntropyWithLogitsLoss

from src.optimizer.adam import AdamW


def cnn_init(processed_path: Path | str, conv_layer_params: list[tuple[int, int, int]], seed: int | None = None) -> dict:
    """
    This method initializes the CNN by creating an instance of every module in the network and creating the train, validation, and test set.

    Args:
        processed_path (Path | str): The path to the .npz-file containing the processed data.
        conv_layer_params (list[tuple[int, int, int]]): The parameters for each convolution layer in the network (num_filters, in_channels, kernel_width(=kernel_height))
        seed (int | None, optional): A random seed used for reproducibility. Default is `None`.
    
    Returns:
        A dictonary containing all the important module instances and the dataloader for all sets.
    """
    data_manager = DataManager(processed_path, seed=seed)
    train_loader, val_loader, _ = data_manager.get_loader()

    seed = None if seed is None else seed + 1

    model = MnistCNN(conv_layer_params, seed=seed)

    crossEntropyWithLogitsLoss = CrossEntropyWithLogitsLoss()

    optimizer = AdamW(model.parameters())

    return dict(train_loader=train_loader, val_loader=val_loader, model=model, crossEntropyWithLogitsLoss=crossEntropyWithLogitsLoss, optimizer=optimizer)


def evalute(val_loader: Dataloader, model: MnistCNN, lossfn: CrossEntropyWithLogitsLoss, n_classes: int = 10, eps: float = 1e-8) -> list[tuple[str, float]]:
    """
    This method evaluates the validation set on the model to calculate performace metrics.

    Args:
        val_loader (Dataloader): The dataloader containing the features and labels of the validation set.
        model (MnistCNN): The trained CNN model.
        lossfn (CrossEntropyWithLogitsLoss): The loss function used in this model.
        n_classes (int, optional): The number of different classes contained in the truth labels. Default is `10` for MNIST.
        eps (float, optional): A factor used to avoid division by zero. 
    
    Returns:
        A list of tuples containg the name of metric and its value.
    """
    metrics = []

    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

    n_batches = len(val_loader)

    total_loss = 0.0
    total_accuracy = 0.0
    total_precision = np.zeros(n_classes, dtype=np.float32)
    total_recall = np.zeros(n_classes, dtype=np.float32)
    total_f1 = np.zeros(n_classes, dtype=np.float32)

    for X, y in val_loader:
        logits = model(X)
        total_loss += lossfn(logits, y)

        softmax = math_fn.softmax(logits)

        preds = np.argmax(softmax, axis=1)

        np.add.at(confusion_matrix, (y, preds), 1)

        tp = np.diag(confusion_matrix)

        total_accuracy += np.sum(tp) / (np.sum(confusion_matrix) + eps)

        precision = tp / (np.sum(confusion_matrix, axis=0) + eps)
        total_precision += precision

        recall = tp / (np.sum(confusion_matrix, axis=1) + eps)
        total_recall += recall

        total_f1 += 2 * precision * recall / (precision + recall + eps)

    metrics.append(("loss", total_loss / n_batches))
    metrics.append(("accuracy", total_accuracy / n_batches))
    metrics.append(("precision", np.mean(total_precision)))
    metrics.append(("recall", np.mean(total_recall)))
    metrics.append(("f1", np.mean(total_f1)))

    return metrics
