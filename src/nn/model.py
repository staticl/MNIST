import numpy as np

from .module import Module
from .sequential import Sequential

from .conv2D import Conv2D
from .activation import ReLU
from .maxpool import MaxPool
from .linear import Linear


class MnistCNN(Module):
    def __init__(self, conv_layer_params: list[tuple[int, int, int]], img_size: int = 28, pool_kernel_size: int = 2, seed: int | None = None) -> None:
        """
        This class creates the blueprint of the CNN by initializing every layer in the network.

        Args:
            conv_layer_params (list[tuple[int, int, int]]): The parameters for each convolution layer in the network (num_filters, in_channels, kernel_width(=kernel_height))
            img_size (int, optional): The size (=height=width) of an image in the MNIST dataset. Default is `28`.
            pool_kernel_size (int, optional): The size of the kernel used in the max pooling layer. Dafualt is `2`.
            seed (int | None, optional): A random seed used for reproducibility. Default is `None`.
        """
        super().__init__()

        for i, conv_params in enumerate(conv_layer_params):
            seed = None if seed is None else seed + 1
            num_filters, in_channels, kernel_width = conv_params
            self._modules[f"conv{i + 1}"] = Conv2D(num_filters, in_channels, kernel_width, seed=seed)
            self._modules[f"relu{i + 1}"] = ReLU()
            if i < len(conv_layer_params) - 1:
                self._modules[f"maxpool{i + 1}"] = MaxPool(kernel_size=pool_kernel_size)

        seed = None if seed is None else seed + 1
        dims = int(img_size / (pool_kernel_size**(len(conv_layer_params) - 1)))
        self._modules[f"linear1"] = Linear(out_channels=conv_layer_params[-1][0], dims=dims, seed=seed)

        self.net = Sequential(list(self._modules.values()))

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Passes the forward pass of the network to the sequential.

        Args:
            input (np.ndarray): The input that is passed through the network.

        Returns:
            The final output of the last layer of the network.
        """
        return self.net.forward(input)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Passes the backward pass of the network to the sequential.

        Args:
            dout (np.ndarray): The cotangent of the loss function.

        Returns:
            The final cotangent of the last layer of the network.
        """
        return self.net.backward(dout)
