from typing import Optional, Tuple

import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


class TimeWindows(BaseTransform):
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

        tw_early, tw_late, service_time = data.client_time[:,0].view(-1, 1), data.client_time[:,1].view(-1, 1), data.client_time[:,1].view(-1, 1)

        max_cap = data.depot_pos[2]
        if self.norm and tw_early.numel() > 0 and max_cap > 0:

            length = self.interval[1] - self.interval[0]
            tw_early = length * (tw_early / max_cap) + self.interval[0]
            tw_late = length * (tw_late / max_cap) + self.interval[0]
            service_time = length * (service_time / max_cap) + self.interval[0]

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.x = torch.cat([pseudo, tw_early.type_as(pseudo), tw_late.type_as(pseudo), service_time.type_as(pseudo)], dim=-1)
        else:
            data.x = torch.cat([tw_early.type_as(pseudo), tw_late.type_as(pseudo), service_time.type_as(pseudo)], dim=-1)

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')