import torch
import numpy as np
import pandas as pd
from typing import List
from torch.utils import data
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, random_split


class TabularRegression(data.Dataset):
    def __init__(self, path: str, task_names: List[str], split_ratio: float = None,
                 split: str = "train", stratify: bool = False):
        """
        Read any tabular dataset (with the last column being labels) as PyTorch TensorDataset
        :param stratify: whether to stratify the dataset while splitting
        :param path: path to the csv file
        :param split_ratio: Train/Validation split ratio
        :return: Train and validation dataset if split_ratio is define, the entire dataset otherwise
        """
        if split_ratio:
            assert split_ratio >= 0
            assert split_ratio <= 1

        self.task_names = task_names

        df = pd.read_csv(path, index_col=False)
        # df = df.fillna(0)
        df = df.fillna(-999)

        x_arr = df.iloc[:, :2151].to_numpy()
        y_arr = df[self.task_names].to_numpy()

        x_tensor = torch.tensor(x_arr, dtype=torch.float32)  # .to_numpy()
        x_tensor = x_tensor.unsqueeze(1)  # Add channel dimension -> (N, C, seq_len)

        y_tensor = torch.tensor(y_arr, dtype=torch.float32)  # .values
        dataset = TensorDataset(x_tensor, y_tensor)

        if "Train" in df.columns:
            print("Using fixed splits")
            train_set, val_set = TensorDataset(*dataset[df["Train"]]), TensorDataset(*dataset[~df["Train"]])
            self.train_set = train_set
            self.val_set = val_set

        if split_ratio and "Train" not in df.columns:
            train_size = int(len(dataset) * split_ratio)
            val_size = len(dataset) - train_size
            if stratify:
                bin_edges = np.histogram_bin_edges(y_tensor, bins="auto")
                print(bin_edges)
                y_binned = np.digitize(y_tensor, bin_edges, right=True)
                train_set, val_set = train_test_split(
                    dataset, train_size=train_size, test_size=val_size, random_state=0,
                    shuffle=True, stratify=y_binned
                )
            else:
                train_set, val_set = random_split(
                    dataset=dataset, lengths=[train_size, val_size], generator=torch.manual_seed(0)
                )
            self.train_set = train_set
            self.val_set = val_set

        else:
            self.train_set = None
            self.val_set = None

        if split == "train":
            self.dataset = self.train_set
        else:
            self.dataset = self.val_set

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):  # -> T_co
        x, y = self.dataset[index]
        sample = {task_name: y[i] for i, task_name in enumerate(self.task_names)}
        sample["image"] = x

        return sample
