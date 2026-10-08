import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .module import Module


class Conv2D(Module):
    def __init__(self, num_filters: int, in_channels: int, kernel_width: int, seed: int | None = None) -> None:
        """
        This class handles the initialization and the forward and backward pass of the 2D convolution layer.

        Args:
            num_filters (int): The number of filters going of the data.
            in_channels (int): The number of input channels (Is `1` for the first layer)
            kernel_width: (=kernel_height) The width and height of the kernel sliding over the data
            seed (int | None, optional): A random seed used for reproducibility. Default is `None`.
        """
        super().__init__()
        rng = np.random.default_rng(seed)

        self.num_filters = num_filters
        self.in_channles = in_channels
        self.kernel_width = kernel_width
        
        self._parameters["W"] = rng.normal(loc=0, scale=1, size=(num_filters, in_channels, kernel_width, kernel_width)).astype(np.float32)
        self._parameters["b"] = np.zeros((num_filters, 1, 1), dtype=np.float32)

        self._gradients["W"] = np.zeros_like(self._parameters["W"], dtype=np.float32)
        self._gradients["b"] = np.zeros_like(self._parameters["b"], dtype=np.float32)

    def set_parameters(self, W: np.ndarray, b: np.ndarray) -> None:
        """
        This function is used to set values the weight and bias to a specific value. This can be used when loading parameters from a previously trained model.

        Args:
            W (np.ndarray): weight matrix (shape = (num_filters, in_channels, kernel_width, kernel_width))
            b (np.ndarray): bias vector (shape = (num_filters, 1, 1))
        
        Raises:
            ValueError: If either the weight or bias has a different shape than it is supposed to.
        """
        if (self._parameters["W"].shape != W.shape) or (self._parameters["b"].shape != b.shape):
            raise ValueError("Shape mismatch in either the weight matrix or the bias vector.")

        self._parameters["W"] = W
        self._parameters["b"] = b

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        This method calculates the forward pass of the 2D convolution layer.

        y = X @ W + b

        Args:
            X (np.ndarray): The input feature matrix in the calculation (shape = (batch_size, height, width))
        
        Returns:
            The predicted logits of this layer y (shape = (batch_size, num_filters, height, width)).
        
        Raises:
            ValueError: If there is a shape mismatch in X.
        """
        batch_size, height, width = X.shape

        if height != width: # this should never be the case for the images in the MNIST dataset
            raise ValueError(f"Shape Error in the input matrix X. {height} (height) != {width} (width)")
        
        pad_width = self.kernel_width - 1
        self.left = int(pad_width / 2)
        self.right = int(pad_width / 2) + pad_width % 2
        padding = ((0, 0), (self.left, self.right), (self.left, self.right))

        X_padded = np.pad(X, padding, mode='constant', constant_values=0)

        X_view = sliding_window_view(X_padded, window_shape=(self.kernel_width, self.kernel_width), axis=(1,2))
        # X_view.shape = (batch_size, height, width, kernel_width, kernel_width)

        output = np.einsum('bhwkk,fckk->bfhw', X_view, self._parameters["W"], optimize=True) + self._parameters["b"]

        return output

    def backward(self) -> np.ndarray:
        pass