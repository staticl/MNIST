import numpy as np


class Module:
    def __init__(self) -> None:
        self._parameters = {}
        self._modules = {}

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self) -> list[np.ndarray] | list[list[np.ndarray]]:
        all_params = []

        for param in self._parameters:
            all_params.extend(param)

        for module in self._modules:
            if hasattr(module, "parameters"):
                all_params.extend(module.parameters())

        return all_params