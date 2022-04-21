from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils import data
from torch.utils.data import random_split, TensorDataset
from torch.utils.data.dataset import T_co


class TabularRegression(data.Dataset):
    def __init__(self, path: str, task_names: List[str], split_ratio: float = None, seq_len: int = 1901,
                 split: str = "train", stratify: bool = False):
        assert split == "train" or split == "val"

        if split_ratio:
            assert split_ratio >= 0
            assert split_ratio <= 1

        self.task_names = task_names

        df = pd.read_csv(path)
        x_arr = df.iloc[:, :seq_len]
        y_arr = df[self.task_names]

        x_tensor = torch.tensor(x_arr.to_numpy(), dtype=torch.float32)
        x_tensor = x_tensor.unsqueeze(1)  # Add channel dimension -> (N, C, seq_len)
        y_tensor = torch.tensor(y_arr.to_numpy(), dtype=torch.float32)
        dataset = TensorDataset(x_tensor, y_tensor)

        if "Train" in df.columns:
            self.train_set, self.val_set = TensorDataset(*dataset[df["Train"]]), TensorDataset(*dataset[~df["Train"]])
        elif split_ratio:
            train_size = int(len(dataset) * split_ratio)
            val_size = len(dataset) - train_size
            if stratify:
                bin_edges = np.histogram_bin_edges(y_tensor, bins="auto")
                print(bin_edges)
                y_binned = np.digitize(y_tensor, bin_edges, right=True)
                self.train_set, self.val_set = train_test_split(
                    dataset, train_size=train_size, test_size=val_size, random_state=0,
                    shuffle=True, stratify=y_binned
                )
            else:
                self.train_set, self.val_set = random_split(
                    dataset=dataset, lengths=[train_size, val_size], generator=torch.manual_seed(0)
                )
        else:
            self.train_set = None
            self.val_set = None

        if split == "train":
            self.dataset = self.train_set
        else:
            self.dataset = self.val_set

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> T_co:
        x, y = self.dataset[index]
        sample = {task_name: y[i] for i, task_name in enumerate(self.task_names)}
        sample["image"] = x

        return sample
