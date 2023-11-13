from typing import Optional, Tuple

import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


class Demand(BaseTransform):
    r""" Each dermand gets globally normalized to a specified interval (:math:`[0, 1]` by default).

    Args:
        max_value (float, optional): If set and :obj:`norm=True`, normalization
            will be performed based on this value instead of the maximum value
            found in the data. (default: :obj:`None`)
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
        interval ((float, float), optional): A tuple specifying the lower and
            upper bound for normalization. (default: :obj:`(0.0, 1.0)`)
    """
    def __init__(
            self,
            norm: bool = True,
            cat: bool = True,
            interval: Tuple[float, float] = (0.0, 1.0),
    ):
        self.norm = norm
        self.cat = cat
        self.interval = interval

    def forward(self, data: Data) -> Data:
        pseudo = data.x

        demand, capacity = data.client_demand[:,0].view(-1, 1), data.client_demand[:,1].view(-1, 1)

        max_cap = capacity.max()
        if self.norm and demand.numel() > 0:

            length = self.interval[1] - self.interval[0]
            demand = length * (demand / max_cap) + self.interval[0]
            capacity = length * (capacity / max_cap) + self.interval[0]

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.x = torch.cat([pseudo, demand.type_as(pseudo), capacity.type_as(pseudo)], dim=-1)
        else:
            data.x = demand

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')