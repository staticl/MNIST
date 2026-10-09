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

        self.X = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        This method calculates the forward pass of the 2D convolution layer.

        `y = X @ W + b`

        Args:
            input (np.ndarray): The input feature matrix in the calculation (shape = (batch_size, in_channels, height, width)).
        
        Returns:
            The predicted logits of this layer y (shape = (batch_size, num_filters, height, width)).
        
        Raises:
            ValueError: If there is a shape mismatch in X.
        """
        _, _, height, width = input.shape

        if height != width: # this should never be the case for the images in the MNIST dataset
            raise ValueError(f"Shape Error in the input matrix X. {height} (height) != {width} (width)")
        
        pad_width = self.kernel_width - 1
        self.left = int(pad_width / 2)
        self.right = int(pad_width / 2) + pad_width % 2
        padding = ((0, 0), (0, 0), (self.left, self.right), (self.left, self.right))

        X_padded = np.pad(input, padding, mode="constant", constant_values=0)

        self.X = sliding_window_view(X_padded, window_shape=(self.kernel_width, self.kernel_width), axis=(2,3))
        # X.shape = (batch_size, height, width, kernel_width, kernel_width)

        output = np.einsum("bchwij,fcij->bfhw", self.X, self._parameters["W"], optimize=True) + self._parameters["b"]

        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        This method calculates the backward pass of 2D convolution layer.

        It calculates the cotangent for this layer, `dX = dout @ W`, and the gradient of the bias and the weights

        Args:
            dout (np.ndarrray): The cotangent of the previous layers in the backward pass (shape = (batch_size, num_filters, height, width))
        
        Returns:
            The cotangent of this layer dX (shape = (batch_size, height, width))
        
        Raises:
            RunTimeError: If the forward pass wasn't called before hand.
        """
        if self.X is None:
            raise RuntimeError(f"The forward pass hasn't been executed.")

        dW = np.einsum("bfhw,bchwij->fcij", dout, self.X, optimize=True)
        self._gradients["W"][:] = dW

        db = np.sum(dout, axis=(0, 2, 3)).reshape(-1, 1, 1)
        self._gradients["b"][:] = db

        padding = ((0, 0), (0, 0), (self.right, self.left), (self.right, self.left)) # padding reversed to forward pass
        dout_padded = np.pad(dout, padding, mode="constant", constant_values=0)
        dout_view = sliding_window_view(dout_padded, window_shape=(self.kernel_width, self.kernel_width), axis=(2,3))

        dX = np.einsum("bfhwij,fcij->bchw", dout_view, self._parameters["W"][:,:,::-1,::-1], optimize=True)

        return dX


