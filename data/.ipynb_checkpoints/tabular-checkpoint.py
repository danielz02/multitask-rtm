from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils import data
from torch.utils.data import random_split, TensorDataset
from torch import nn

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, TensorDataset, random_split, DataLoader




class TabularRegression(data.Dataset):
    def __init__(self, path: str, task_names: List[str], split_ratio: float = None, seq_len: int = 1901,
                 split: str = "train", stratify: bool = False):
        """
        Read any tabular dataset (with the last column being labels) as PyTorch TensorDataset
        :param stratify: whether to stratify the dataset while splitting
        :param scale: whether to scale data
        :param classification: whether to output classification labels for SOC
        :param target_name: the name of the label column in the CSV
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

        x_tensor = torch.tensor(x_arr, dtype=torch.float32) #.to_numpy()
        x_tensor = x_tensor.unsqueeze(1)  # Add channel dimension -> (N, C, seq_len)

        y_tensor = torch.tensor(y_arr, dtype=torch.float32) #.values
        dataset = TensorDataset(x_tensor, y_tensor)

        if "Train" in df.columns:
            print("Using fixed splits")
            train_set, val_set = TensorDataset(*dataset[df["Train"]]), TensorDataset(*dataset[~df["Train"]])
            return train_set, val_set
        if split_ratio:
            train_size = int(len(dataset) * split_ratio)
            val_size = len(dataset) - train_size
            if stratify:
                bin_edges = np.histogram_bin_edges(y_tensor, bins="auto")
                # bin_edges = list(range(1, 13))
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
        #self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index): # -> T_co
        x, y = self.dataset[index]
        sample = {task_name: y[i] for i, task_name in enumerate(self.task_names)}
        sample["image"] = x

        return sample



# class TabularRegression(data.Dataset):
#     def __init__(self, path: str, task_names: List[str], split_ratio: float = None, seq_len: int = 1901,
#                  split: str = "train", stratify: bool = False):
#         assert split == "train" or split == "val"

#         if split_ratio:
#             assert split_ratio >= 0
#             assert split_ratio <= 1

#         self.task_names = task_names

#         df = pd.read_csv(path)
#         x_arr = df.iloc[:, :seq_len]
#         y_arr = df[self.task_names]

#         x_tensor = torch.tensor(x_arr.to_numpy(), dtype=torch.float32)
#         x_tensor = x_tensor.unsqueeze(1)  # Add channel dimension -> (N, C, seq_len)
#         y_tensor = torch.tensor(y_arr.to_numpy(), dtype=torch.float32)
#         dataset = TensorDataset(x_tensor, y_tensor)

#         if "Train" in df.columns:
#             self.train_set, self.val_set = TensorDataset(*dataset[df["Train"]]), TensorDataset(*dataset[~df["Train"]])
#         elif split_ratio:
#             train_size = int(len(dataset) * split_ratio)
#             val_size = len(dataset) - train_size
#             if stratify:
#                 bin_edges = np.histogram_bin_edges(y_tensor, bins="auto")
#                 print(bin_edges)
#                 y_binned = np.digitize(y_tensor, bin_edges, right=True)
#                 self.train_set, self.val_set = train_test_split(
#                     dataset, train_size=train_size, test_size=val_size, random_state=0,
#                     shuffle=True, stratify=y_binned
#                 )
#             else:
#                 self.train_set, self.val_set = random_split(
#                     dataset=dataset, lengths=[train_size, val_size], generator=torch.manual_seed(0)
#                 )
#         else:
#             self.train_set = None
#             self.val_set = None

#         if split == "train":
#             self.dataset = self.train_set
#         else:
#             self.dataset = self.val_set

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index) -> T_co:
#         x, y = self.dataset[index]
#         sample = {task_name: y[i] for i, task_name in enumerate(self.task_names)}
#         sample["image"] = x

#         return sample
