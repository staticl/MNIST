"""
This is a test script used to validate the dataloaders gotten from the 'Datamanager' class.

Run this script via: `python -m tests.test_datamanager` 
"""

import numpy as np

import os
from pathlib import Path 

from src.data.datamanager import DataManager


# seed for reproducibility
SEED = 101

# default rng for reproducibility across test attempts
rng = np.random.default_rng(seed=SEED)

# random generated input feature matrix
PATH_TO_PROCESSED = Path(r"data\mnist_preprocessed.npz")

def test_dataloader() -> None:
    datamanager = DataManager(PATH_TO_PROCESSED, batch_size=32, seed=SEED)
    train_loader, val_loader, test_loader = datamanager.get_loader()

    X_shape = None
    y_shape = None

    for X_batch, y_batch in train_loader:
        if X_shape is None or y_shape is None:
            X_shape = X_batch.shape
            y_shape = y_batch.shape
        if X_batch.shape != X_shape or y_batch.shape != y_shape:
            raise ValueError(f"There is a shape mismatch in either X or y: X_batch.shape = {X_batch.shape}, y_batch.shape = {y_batch.shape}")

    print(f"X_batch.shape = {X_shape}, y_batch.shape = {y_shape}")
    # X_batch.shape = (batch_size, in_channels, height, width), y_batch.shape = (batch_size, in_channels)

if __name__ == "__main__":
    test_dataloader()