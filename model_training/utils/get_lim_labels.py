from typing import Tuple

import torch
from torch import Tensor


def get_lim_labels(label: Tensor, shape: Tuple[int, int, int]) -> Tensor:
    max_move, p1, p2 = shape
    list_test = []
    copy = label.clone()
    copy = copy.reshape(shape)
    for i in range(max_move):
        for i2 in range(p1):
            if i2 < p2:
                list_test.append([i, i2, i2])
            else:
                list_test.append([i, i2, 0])

    indices = torch.tensor([list_test])
    result = copy[indices[:, :, 0], indices[:, :, 1], indices[:, :, 2]]
    return result.flatten()
