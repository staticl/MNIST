"""
This is a test script used to validate the forward and backward pass of the linear layer.

Run this script via: `python -m tests.test_linear` 
"""

import numpy as np

import os
from pathlib import Path 

from src.nn.linear import Linear


# seed for reproducibility
SEED = 101

# default rng for reproducibility across test attempts
rng = np.random.default_rng(seed=SEED)

# random generated input feature matrix
input = rng.normal(loc=0, scale=1, size=(32, 16, 28, 28)).astype(np.float32) # shape = (batchsize=32, out_channels=16, height=28, width=28)
dout = rng.normal(loc=0, scale=1, size=(32, 10)).astype(np.float32) # shape = (batchsize=32, n_classes=10)

lin_layer = Linear(16, seed=SEED + 1) # num_filters=16, input_channels=1, kernel_width=5

def test_linear_forward() -> None:
    out = lin_layer.forward(input)
    print(f"out.shape = {out.shape}, max(out) = {np.max(out)}, min(out) = {np.min(out)}")

def test_linear_backward() -> None:
    out = lin_layer.backward(dout)
    print(f"out.shape = {out.shape}, max(out) = {np.max(out)}, min(out) = {np.min(out)}")

if __name__ == "__main__":
    test_linear_forward()
    test_linear_backward()