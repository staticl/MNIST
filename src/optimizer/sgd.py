import numpy as np


class SGD:
    def __init__(self, parameters: list[tuple[np.ndarray, np.ndarray]], lr: float = 1e-3) -> None:
        self.parameters = parameters

        print(f"Number-Parameters: {len(self.parameters)}")

        self.lr = lr

    def step(self):
        for param, grad in self.parameters:
            clipped_grad = np.clip(grad, -1.0, 1.0)
            param -= self.lr * clipped_grad
