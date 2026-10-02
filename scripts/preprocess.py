from data.preprocess import ImageProcessing

import os
from pathlib import Path

def main() -> None:
    cwd = Path.cwd()
    mnist_folder_path = os.path.join(cwd, Path(r"data\mnist-dataset"))

    mnist_arr_path = os.path.join(cwd, Path(r"data\mnist_data_arr.npz"))

    imgproc = ImageProcessing(mnist_folder_path, mnist_arr_path)


if __name__ == "__main__":
    main()