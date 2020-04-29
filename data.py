import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def gau(grid):
    return torch.exp(-(grid[0] ** 2 + grid[1] ** 2) / 2)


class MyDataset(torch.utils.data.Dataset):
    """Some Information about MyDataset"""

    def __init__(self):
        super(MyDataset, self).__init__()
        x = torch.arange(-4, 4, 0.1)
        _y, _x = torch.meshgrid(x, x)
        grid = torch.stack([_x, _y], 0)
        grid = grid.reshape(2, -1).transpose(1, 0)
        self.grid = grid

    def __getitem__(self, index):
        feature = self.grid[index]
        label=gau(feature)
        return feature, label

    def __len__(self):
        return len(self.grid)


if __name__ == "__main__":
    dataset=MyDataset()
    dataset[0]
