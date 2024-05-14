from typing import Union

from torch_geometric.data import Data

from .Demand import Demand
from .Distance import Distance
from .NodePosition import CenterScale
from .TimeWindows import TimeWindows


def normalize_graphs(
    data: Data, max_distance: Union[int, float] = None, TW: bool = False
) -> Data:
    if max_distance:
        norm_distance = Distance(max_value=max_distance, cat=False)
    else:
        norm_distance = Distance(cat=False)
    data = norm_distance(data)

    norm_demand = Demand()
    data = norm_demand(data)

    pos_scale = CenterScale()
    data = pos_scale(data)

    if TW:
        scaleTime = TimeWindows()
        data = scaleTime(data)

    return data
