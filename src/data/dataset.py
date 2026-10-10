import numpy as np

from typing import SupportsIndex


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This class provides an abstract `Dataset` object.

        Args:
            X (np.ndarray): The feature matrix of shape (n_set_samples, in_channels, height, width) with values between [-1, 1].
            y (np.ndarray): The value matrix of shape (n_set_samples,) with values in {0, 1, ..., 9}.
        """
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """
        Returns:
            The number of samples inside the `Dataset` object.
        """
        return len(self.X)

    def __getitem__(self, idx: SupportsIndex) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            Batches of the feature matrix and value vector at the index positions.
        """
        return self.X[idx], self.y[idx]