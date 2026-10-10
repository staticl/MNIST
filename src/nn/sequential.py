import numpy as np

from .module import Module


class Sequential(Module):
    def __init__(self, layers: list[Module]) -> None:
        """
        This class manages the forward and backward pass across all layers.

        Args:
            layers (list[Module]): A list of every layer inside the CNN.
        """
        super().__init__()
        self.layers = layers

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        This method passes the input through all layers of the network.

        Args:
            input (np.ndarray): The input that is passed through the network.
        
        Returns:
            The final output of the last layer of the network.
        """
        for layer in self.layers:
            input = layer.forward(input)

        return input

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        This method passes the cotangent of the loss in reverse order through all layers of the network.
        
        Args:
            dout (np.ndarray): The cotangent of the loss function.
        
        Returns:
            The final cotangent of the last layer of the network.
        """
        for layer in self.layers[::-1]:
            dout = layer.forward(dout)

        return dout