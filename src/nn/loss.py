import numpy as np

from utils import math_fn


class CrossEntropyWithLogitsLoss:
    def __init__(self):
        self.logits = None
        self.targets = None

    def __call__(self, logits: np.ndarray, true: np.ndarray) -> float:
        """
        Makes the class object itself callable. Passes the call to `forward()`.
        """
        return self.forward(logits, true)
    
    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        This method calculates the cross-entropy loss from the raw logits.

        Args:
            logits (np.ndarray): The logits outputted by the last convolutional layer of the CNN (shape = (batch_size, n_classes=10)).
            targets (np.ndarray): The truth labels containing the correct class labels (shape = (batch_size,))
        
        Returns:
            The calculated cross-entropy loss.
        """
        self.logits = logits
        self.targets = targets

        self.batch_size = len(self.targets)

        max_logits = np.max(self.logits, axis=-1, keepdims=True)

        log_softmax = (self.logits - max_logits) - np.log(np.sum(np.exp(self.logits - max_logits), axis=-1, keepdims=True))

        loss = -log_softmax[np.arange(self.batch_size), self.targets]

        return np.mean(loss)

    def backward(self) -> np.ndarray:
        """
        This method calculates the backward pass of the loss.

        It calculates the cotangent for this layer, `dL = softmax(z) - y_onehot`.
        
        Returns:
            The cotangent of this layer dL (shape = (batch_size, n_classes)) and the softmax of the logits.
        
        Raises:
            RunTimeError: If the forward pass wasn't called before hand.
        """
        if (self.logits is None) or (self.targets is None):
            raise RuntimeError("The forward pass hasn't been executed prior.")

        softmax = math_fn.softmax(self.logits)

        grad = softmax.copy()
        grad[np.arange(self.batch_size), self.targets] -= 1

        return grad / self.batch_size
