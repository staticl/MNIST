from .module import Module
from .sequential import Sequential

class MnistCNN(Module):
    def __init__(self, conv_layer_hyparams: list[tuple[int, int, int]], seed: int | None = None) -> None:
        super().__init__()