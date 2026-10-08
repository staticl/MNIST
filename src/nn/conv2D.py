import numpy as np

from module import Module


class Conv2D(Module):
    def __init__(self, num_filters: int, in_channels: int, kernel_width: int, seed: int | None = None) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        
        self._parameters["W"] = rng.normal(loc=0, scale=1, size=(num_filters, in_channels, kernel_width)).astype(np.float32)
        self._parameters["b"] = np.zeros((num_filters, 1), dtype=np.float32)

    def set_parameters(self, W: np.ndarray, b: np.ndarray) -> None:
        self._parameters["W"] = W
        self._parameters["b"] = b

    def forward(self) -> np.ndarray:
        pass

    def backward(self) -> np.ndarray:
        pass