from typing import Union, Tuple

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


class Center(BaseTransform):
    """Centers node positions :obj:`data.pos` around the origin.
       the origin is defined as the position of Depot
    """

    def forward(
            self,
            data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            if hasattr(store, 'pos'):
                store.pos = store.pos - data.depot_pos
        return data


class CenterScale(BaseTransform):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`
    (functional name: :obj:`normalize_scale`).
    """

    def __init__(self):
        self.center = Center()

    def forward(self, data: Data) -> Data:
        data = self.center(data)
        pseudo = data.x
        scale = (1 / data.pos.abs().max()) * 0.999999
        scaled_pos = data.pos * scale

        if pseudo is not None:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.x = torch.cat([pseudo, scaled_pos.type_as(pseudo)], dim=-1)

        return data
