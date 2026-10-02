import numpy as np


class Module:
    def __init__(self) -> None:
        self._parameters = {}
        self._modules = {}

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @property
    def parameters(self) -> list[np.ndarray] | list[list[np.ndarray]]:
        all_params = []

        for param in self._parameters:
            all_params.append(param)

        for module in self._modules:
            if hasattr(module, "parameter"):
                all_params.append(module.parameters)

        return all_params