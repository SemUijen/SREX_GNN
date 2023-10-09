import math
from typing import Tuple

import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from data.utils.ParentGraphDataset import ParentGraphsDataset


def get_train_test_loader(dataset: ParentGraphsDataset, seed: int = 42, batchsize: int = 1, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:

    size = len(dataset)
    train_size = math.floor(0.8 * size)
    test_size = size - train_size

    generator1 = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [train_size, test_size], generator=generator1)

    train_loader = DataLoader(dataset=train_set, batch_size=batchsize, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_set, batch_size=batchsize, num_workers=num_workers)
    return train_loader, test_loader
