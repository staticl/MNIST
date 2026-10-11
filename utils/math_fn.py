import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    max_logits = np.max(logits, axis=-1, keepdims=True)
    return np.exp(logits - max_logits) / np.sum(np.exp(logits - max_logits), axis=-1, keepdims=True)
