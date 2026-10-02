import numpy as np

import os
import shutil
from pathlib import Path

import kagglehub


class ImageProcessing:
    def __init__(self, mnist_dataset_path: Path | str, save_preprocess_path: Path | str, kaggleset_mnist_path: Path | str = "scolianni/mnistasjpg") -> None:
        """
        This class transforms the raw MNIST image data into tensors of shape (1, n_pixels, n_pixels). It stacks all tensors of each image into a single numpy.ndarray.
        The array gets saved to a '.npy'-file.

        Args:
            mnist_dataset_path (Path | str): The path to the folder where the mnist-dataset will be saved to.
            save_preprocess_path (Path | str): The path to the file containing the MNIST data array.
            kaggleset_mnist_path (Path | str, optional): The path to the mnist-dataset as '.jpg'-format. Default here is: 'scolianni/mnistasjpg'.

        Raises:
            ValueError: If the file at 'save_preprocess_path' already exists.
        """
        if os.path.exists(save_preprocess_path):
            raise ValueError(f"The processed data already exists at: {save_preprocess_path}. Please delete this file if the preprocessing should be redone.")
        self.save_preprocess_path = save_preprocess_path

        if not os.listdir(mnist_dataset_path):
            cache = kagglehub.dataset_download(str(kaggleset_mnist_path))
            shutil.move(cache, mnist_dataset_path)

        self.mnist_dataset_path = os.path.join(mnist_dataset_path, r"training_set/training_set")

    def load_mnist(self, augment: bool = True, seed: int | None = None) -> None:
        """
        This method converts the MNIST-dataset into a single tensor which can be saved as a binary file.

        Args:
            augment (bool, optional): If set to `True` some of the data will get random augmentations. The default is set to `True`.
            seed (int | None, optional): A random seed, used for reproducibility.
        """
        self.rng = np.random.default_rng(seed)

    def _mnist_image_loader(self) -> tuple[np.ndarray, float]:
        pass

    def _augment(self) -> None:
        pass

    def _convert_to_tensor(self) -> None:
        pass