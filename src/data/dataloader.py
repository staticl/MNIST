import numpy as np

from typing import Self

from .dataset import Dataset


class Dataloader:
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = False, seed: int | None = None) -> None:
        """
        This class creates a 'Dataloader' object that is used to iterate over the dataset. The iterator yields a batch of image matrices. 

        Args:
            X (np.ndarray): The feature matrix of shape (n_set_samples, in_channels, height, width) with values between [-1, 1].
            y (np.ndarray): The value matrix of shape (n_set_samples,) with values in {0, 1, ..., 9}.
            batch_size (int, optional): The size of a single batch returned by the dataloader. Default is `32`.
            shuffle (bool, optional): Decides weather a set gets shuffled or not (used for training set). Default is `False`.
            seed (int | None, optional): A random seed used for reproducibility. Default is `None`.
        """
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        self.batch_size = batch_size

        surplus = len(X) % batch_size

        start = int(surplus / 2)
        end = len(X) - (int(surplus / 2) + surplus % 2)
        
        X_trimmed = X[start:end]
        y_trimmed = y[start:end]

        self.n_samples = len(X_trimmed)
        self.indices = np.arange(self.n_samples)
        
        self.dataset = Dataset(X_trimmed, y_trimmed)

    def __len__(self) -> int:
        return (self.n_samples / self.batch_size)

    def __iter__(self) -> Self:
        self.current_idx = 0

        if self.shuffle:
            self.rng.shuffle(self.indices)

        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        if self.current_idx >= self.n_samples:
            raise StopIteration

        start = self.current_idx
        end = self.current_idx + self.batch_size

        self.current_idx += self.batch_size

        return self.dataset[self.indices[start:end]]         