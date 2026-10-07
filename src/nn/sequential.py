import numpy as np

from module import Module


class Sequential(Module):
    def __init__(self, layers: list[Module]) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, input: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            input = layer.forward(input)

        return input

    def backward(self, dout: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            dout = layer.forward(dout)

        return dout