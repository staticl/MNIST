import numpy as np

import os
from pathlib import Path

from .dataloader import Dataloader

class DataManager:
    def __init__(self, processed_path: Path | str, batch_size: int = 32, seed: int | None = None) -> None:
        """
        This class manages the data loading. It reads the processed data from .npz-file and initiates the 'Dataloader' objects.

        Args:
            processed_path (Path | str): The path to the .npz-file containing the processed data.
            batch_size (int, optional): The size of a single batch returned by the dataloader. Default is `32`.
            seed (int | None, optional): A random seed used for reproducibility. Default is `None`.
        
        Raises:
            FileNotFoundError: If the file containing the processed data doesn't exist. 
        """ 
        if not os.path.exists(processed_path):
            raise FileExistsError(f"The file containing the preprocessed data doesn't exist. Path: {processed_path}")
        
        feature_arr, target_arr = np.load(processed_path)
        # feature_arr.shape = (n_digits, height, width), target_arr.shape = (n_digits,)

        self.feature_arr = feature_arr.astype(np.float32)
        self.target_arr = target_arr.astype(np.float32)

        self.batch_size = batch_size

        self.seed = seed

    def get_loader(self, train_ratio: float = 0.7) -> tuple[Dataloader, Dataloader, Dataloader]:
        """
        This method splits the dataset into train, validation and test set and creates a 'Dataloader' object for each.

        Args:
            train_ratio (float, optional): The percentage of the dataset used for the training. Default is `0.7`. 
                                           The rest gets equally split into validation and test set.

        Returns:
            A seperate 'Dataloader' object for each of the datasets.
        
        Raises:
            ValueError: If train_ratio is less or equal to `1` or less or equal to `0`.
        """
        if not (0.0 < train_ratio < 1.0):
            raise ValueError("The train ratio has to be in ]0, 1[.")

        n_samples = len(self.dataset)

        train_split = int(n_samples * train_ratio)
        val_split = train_split + int(n_samples * train_ratio / 2)

        X_train, y_train = self.feature_arr[:train_split], self.target_arr[:train_split]
        X_val, y_val = self.feature_arr[train_split:val_split], self.digit_target_arrtensor[train_split:val_split]
        X_test, y_test = self.feature_arr[val_split:], self.target_arr[train_split:val_split]

        train_loader = Dataloader(X_train, y_train, batch_size=self.batch_size, shuffle=True, seed=self.seed)
        val_loader = Dataloader(X_val, y_val, batch_size=self.batch_size)
        test_loader = Dataloader(X_test, y_test, batch_size=self.batch_size)

        return train_loader, val_loader, test_loader


        


        
