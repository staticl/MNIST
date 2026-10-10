import numpy as np


class AdamW:
    def __init__(self, parameters: list[tuple[np.ndarray, np.ndarray]], lr: float = 3e-4, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        """
        This class implements the Adam optimizer. The optimizer uses SGD with momentum and an adaptive learning rate.

        Args:
            parameters (list[tuple[np.ndarray, np.ndarray]]): A list containing the parameter and gradients of every convolution layer inside the CNN.
            lr (float, optional): The learning rate used for the optimizer. Default for adam is `3e-4`.
            beta1 (float, optional): The decay rate for the first-moment estimate.
            beta2 (float, optional): The decay rate for the second-moment estimate.
            eps (float, optional): A factor used to avoid division by zero.
        """
        self.parameters = parameters
        
        self.m = [np.zeros_like(param, dtype=np.float32) for param, _ in parameters]
        self.v = [np.zeros_like(param, dtype=np.float32) for param, _ in parameters]

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.eps = eps

        self.t = 0

    def step(self) -> None:
        """
        This method applies a single step to the optimizer (updates the parameters in place using the gradient).
        """
        self.t += 1
        for param, grad in self.parameters:
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

            m_hat = self.m / (1 - self.beta1**self.t)
            v_hat = self.v / (1 - self.beta2**self.t)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

