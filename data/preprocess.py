import numpy as np

import os
import shutil
from pathlib import Path

import kagglehub
from PIL import Image


class ImageProcessing:
    def __init__(self, mnist_dataset_path: Path | str, save_preprocessed_path: Path | str, kaggleset_mnist_path: Path | str = "scolianni/mnistasjpg") -> None:
        """
        This class transforms the raw MNIST image data into tensors of shape (1, n_pixels, n_pixels). It stacks all tensors of each image into a single numpy.ndarray.
        The array gets saved to a '.npy'-file.

        Args:
            mnist_dataset_path (Path | str): The path to the folder where the mnist-dataset will be saved to.
            save_preprocessed_path (Path | str): The path to the file containing the MNIST data array.
            kaggleset_mnist_path (Path | str, optional): The path to the mnist-dataset as '.jpg'-format. Default here is: 'scolianni/mnistasjpg'.

        Raises:
            FileNotFoundError: If the target directory for the MNIST dataset doesn't exist.
            FileExistsError: If the file at 'save_preprocess_path' already exists.
        """
        if not os.path.exists(mnist_dataset_path):
            raise FileNotFoundError(f"The target directory for the MNIST dataset doesn't exist. Path {mnist_dataset_path}")
        
        if os.path.exists(save_preprocessed_path):
            raise FileExistsError(f"The processed data already exists at: {save_preprocessed_path}. Please delete this file if the preprocessing should be redone.")
        self.save_preprocessed_path = save_preprocessed_path

        if not os.listdir(mnist_dataset_path):
            cache = kagglehub.dataset_download(str(kaggleset_mnist_path))
            shutil.move(cache, mnist_dataset_path)

        self.mnist_dataset_path = os.path.join(mnist_dataset_path, r"trainingSet", r"trainingSet")

        self.n_digits = len(os.listdir(self.mnist_dataset_path))


    def load(self, augment: bool = True, seed: int | None = None) -> None:
        """
        This method converts the MNIST-dataset into a single tensor which can be saved as a binary file.

        Args:
            augment (bool, optional): If set to `True` some of the data will get random augmentations. The default is set to `True`.
            seed (int | None, optional): A random seed, used for reproducibility. Default is `None`.
        """
        self.rng = np.random.default_rng(seed)

        image_arr = []
        digit_arr = []

        for digit in range(self.n_digits):
            digit_path = os.path.join(self.mnist_dataset_path, str(digit))
            images = os.listdir(digit_path)
            for img in images:
                img = Image.open(os.path.join(digit_path, img))
                if augment:
                    img = self._augment(img)
                img_tensor = self._convert_to_tensor(img)
                image_arr.append(img_tensor)
                digit_arr.append(digit)
        
        np.savez(file=self.save_preprocessed_path,
                 feature_arr=np.stack(image_arr), 
                 target_arr=np.array(digit_arr).reshape(-1, 1),
                 allow_pickle=True)

    def _augment(self, img: Image) -> Image:
        """
        Rotates the image between [-20°, 20°].

        Args:
            img (Image): The image as a PIL.image object.
        
        Returns:
            The augmented image with a rotation between [-20°, 20°].
        """
        angle = int(self.rng.normal(loc=0.0, scale=1.0) * 20)
        return img.rotate(angle)

    def _convert_to_tensor(self, img: Image) -> np.ndarray:
        """
        Converts the image to a numpy.ndarray and scales the pixels values to lie between [-1, 1].

        Args:
            img (Image): The image as a PIL.image object.
        
        Returns:
            The image converted to a numpy.ndarray with pixel values scaled between [-1, 1].
        """
        img_tensor = np.array(img) / 127.5 - 1
        
        return img_tensor.reshape(1, 28, 28)
