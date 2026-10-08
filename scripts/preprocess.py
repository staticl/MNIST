"""
This script is utilzed to execute the image preprocessing upfront. The images get converted to tensors and saved to a .npz-file.

This enables a faster training and the ability to handle processing independently.

Run this script via: `python -m scripts.preprocess`
"""

import os
from pathlib import Path

from data.preprocess import ImageProcessing


# path to the raw MNIST image dataset
MNIST_FOLDER_PATH = os.path.join(Path.cwd(), Path(r"data\mnist-dataset"))

# path to the .npz file where the processed data arrays are saved
MNIST_PREPROCESSED_PATH = os.path.join(Path.cwd(), Path(r"data\mnist_preprocessed.npz"))

def main() -> None:
    imgproc = ImageProcessing(MNIST_FOLDER_PATH, MNIST_PREPROCESSED_PATH)
    imgproc.load()

if __name__ == "__main__":
    main()