from torch_geometric.data import Data
from typing import Union

from .Distance import Distance
from .Demand import Demand
from .NodePosition import CenterScale
def normalize_graphs(data: Data, max_distance: Union[int, float] = None) -> Data:
    if max_distance:
        norm_distance = Distance(max_value=max_distance, cat=False)
    else:
        norm_distance = Distance(cat=False)
    data = norm_distance(data)

    norm_demand = Demand()
    data = norm_demand(data)

    pos_scale = CenterScale()
    data = pos_scale(data)

    return data
