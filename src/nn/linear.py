import numpy as np

from .module import Module


class Linear(Module):
    def __init__(self, out_channels: int, dims: int = 28, n_classes: int = 10, seed: int | None = None) -> None:
        """
        This class handles the initialization and the forward and backward pass of the 2D convolution layer.

        Args:
            out_channels (int): The number of filters going of the data.
            dims (int, optional): The height and width of the images in the MNIST dataset. Default is `28`.
            n_classes (int, optional): The number of different digits inside the MNISt dataset. Default is `10`.
            seed (int | None, optional): A random seed used for reproducibility. Default is `None`.
        """
        super().__init__()
        rng = np.random.default_rng(seed)

        self.out_channels = out_channels
        self.dims = dims

        self._parameters["W"] = rng.normal(loc=0.0, scale=1.0, size=(self.out_channels * self.dims**2, n_classes)).astype(np.float32)
        self._parameters["b"] = np.zeros(n_classes, dtype=np.float32)

        self._gradients["W"] = np.zeros_like(self._parameters["W"], dtype=np.float32)
        self._gradients["b"] = np.zeros_like(self._parameters["b"], dtype=np.float32)

        self.X = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        This method calculates the forward pass of the linear layer.

        `y = X @ W + b`

        Args:
            input (np.ndarray): The input feature matrix for this layer (shape = (batch_size, out_channels * height * width)).
        
        Returns:
            The predicted logits of this layer y (shape = (batch_size, n_classes)).
        
        Raises:
            ValueError: If there is a shape mismatch in X.
        """
        batch_size, out_channels, height, width = input.shape

        if out_channels * height * width != self.out_channels * self.dims**2:
            raise ValueError(f"Shape Error in the input matrix X.")

        self.X = input.reshape(batch_size, self.out_channels * self.dims**2)

        output = self.X @ self._parameters["W"] + self._parameters["b"]

        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        This method calculates the backward pass of linear layer.
        
        It calculates the cotangent for this layer, `dX = dout @ W.T`, and the gradient of the bias and the weights.

        Args:
            dout: The cotangent of the previous layers in the backward pass (shape = (batch_size, n_classes)).

        Returns:
            The cotangent of this layer dX (shape = (batch_size, out_channels * height * width)).
        
        Raises:
            RunTimeError: If the forward pass wasn't called before hand.
        """
        if self.X is None:
            raise RuntimeError("The forward pass hasn't been executed prior.")

        dW = self.X.T @ dout
        self._gradients["W"][:] = dW

        db = np.sum(dout, axis=0)
        self._gradients["b"][:] = db

        dX = dout @ self._parameters["W"].T

        return dX.reshape(-1, self.out_channels, self.dims, self.dims)
        