"""
This is a test script used to validate the forward and backward pass of the conv2D layer.

Run this script via: `python -m tests.test_conv2D` 
"""

import numpy as np

from src.nn.conv2D import Conv2D


# seed for reproducibility
SEED = 101

# default rng for reproducibility across test attempts
rng = np.random.default_rng(seed=SEED)

# random generated input feature matrix
X = rng.normal(loc=0, scale=1, size=(32, 1, 28, 28)).astype(np.float32) # shape = (batchsize=32, in_channels=1, height=28, width=28)
dout = rng.normal(loc=0, scale=1, size=(32, 16, 28, 28)).astype(np.float32) # shape = (batchsize=32, num_filters=16, height=28, width=28)

conv_layer = Conv2D(16, 1, 5, seed=SEED + 1) # num_filters=16, input_channels=1, kernel_width=5

def test_conv2D_forward() -> None:
    out = conv_layer.forward(X)
    print(f"out.shape = {out.shape}, max(out) = {np.max(out)}, min(out) = {np.min(out)}")

def test_conv2D_backward() -> None:
    out = conv_layer.backward(dout)
    print(f"out.shape = {out.shape}, max(out) = {np.max(out)}, min(out) = {np.min(out)}")

if __name__ == "__main__":
    test_conv2D_forward()
    test_conv2D_backward()