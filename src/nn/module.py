import numpy as np


class Module:
    def __init__(self) -> None:
        """
        This class functions as a bookkeeping class, keeping track of all parameters and gradients of every layer.
        """
        self._parameters = {}
        self._gradients = {}
        self._modules = {}

    def forward(self, *args, **kwargs):
        """
        Raises:
            NotImplementedError: If the forward pass for layer inside the network hasn't been implemeted.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """
        Makes the class object itself callable.
        """
        return self.forward(*args, **kwargs)

    def parameters(self) -> list[np.ndarray] | list[list[np.ndarray]]:
        """
        Returns:
            A list of all the parameters either of the layer or the whole network.
        """
        all_params = []

        for name, param in self._parameters.items():
            all_params.extend((param, self._gradients[name]))

        for module in self._modules:
            if hasattr(module, "parameters"):
                all_params.extend(module.parameters())

        return all_params