import torch
from torch_geometric.data import Data, Dataset, DataLoader

from typing import Tuple
from torch import Tensor


class FullGraph(Data):
    def __init__(self, edge_index: Tensor, edge_weight: Tensor, client_features: Tensor):
        super().__init__(x=client_features, edge_index=edge_index, edge_attr=edge_weight)


class ParentGraph(Data):

    def __init__(self, client_route_vector: Tensor, edge_index: Tensor, edge_weight:Tensor, num_routes: int, client_features: Tensor):
        super().__init__(x=client_features, edge_index=edge_index, edge_attr=edge_weight)
        self.num_routes = num_routes
        self.client_route_vector = client_route_vector
