import numpy as np

import os
from pathlib import Path

from utils import math_fn

from src.data.datamanager import DataManager
from src.data.dataloader import Dataloader

from src.nn.model import MnistCNN
from src.nn.loss import CrossEntropyWithLogitsLoss

from optimizer.adam import AdamW


def cnn_init(processed_path: Path | str, conv_layer_params: list[tuple[np.ndarray, np.ndarray]], seed: int | None = None) -> dict:
    data_manager = DataManager(processed_path, seed=seed)
    train_loader, val_loader, _ = data_manager.get_loader()
    
    seed = None if seed is None else seed + 1
    
    model = MnistCNN(conv_layer_params, seed)

    crossEntropyWithLogitsLoss = CrossEntropyWithLogitsLoss()

    optimizer = AdamW(model.parameters())

    return dict(train_loader=train_loader, val_loader=val_loader, model=model, crossEntropyWithLogitsLoss=crossEntropyWithLogitsLoss, optimizer=optimizer)

def evalute(val_loader: Dataloader, model: MnistCNN, lossfn: CrossEntropyWithLogitsLoss) -> list[tuple[str, float]]:
    metrics = []
    
    n_batches = len(val_loader)
    total_loss = 0.0
    
    for X, y in val_loader:
        logits = model(X)
        total_loss += lossfn(logits, y)

        softmax = math_fn.softmax(logits)

    metrics.append(("loss", total_loss / n_batches))

    return metrics

    