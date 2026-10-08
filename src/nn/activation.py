import numpy as np

from .module import Module


class ReLU(Module):
    def __init__(self) -> None:
        """
        This class handles the initialization and the forward and backward pass of the ReLU activation function (non-linear layer).

        ReLU(x) = max(0, x)

        ReLU'(x) = 1 if x > 0 else 0
        """
        super().__init__()

        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Returns:
            ReLU(x) = max(0, x)
        """
        self.mask = (x > 0)

        return x * self.mask

    def backward(self, din: np.ndarray) -> np.ndarray:
        """
        Returns:
            dout = din * ReLU'(x)
        """
        return din * self.mask