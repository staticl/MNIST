"""
This is a test script used to validate the forward and backward pass of the loss function.

Run this script via: `python -m tests.test_loss` 
"""

import numpy as np

import os
from pathlib import Path 

from src.nn.loss import CrossEntropyWithLogitsLoss


# seed for reproducibility
SEED = 101

# default rng for reproducibility across test attempts
rng = np.random.default_rng(seed=SEED)

# random generated input feature matrix
targets = (rng.random(size=(32)) * 10).astype(int) # shape = (batchsize=32,)
logits = rng.normal(loc=0, scale=1, size=(32, 10)).astype(np.float32) # shape = (batchsize=32, n_classes=10)

crossEntropyWithLogitsLoss = CrossEntropyWithLogitsLoss()

def test_loss_forward() -> None:
    loss = crossEntropyWithLogitsLoss(logits, targets)
    print(f"loss = {loss}")

def test_loss_backward() -> None:
    out = crossEntropyWithLogitsLoss.backward()
    print(f"out.shape = {out.shape}, max(out) = {np.max(out)}, min(out) = {np.min(out)}")

if __name__ == "__main__":
    test_loss_forward()
    test_loss_backward()