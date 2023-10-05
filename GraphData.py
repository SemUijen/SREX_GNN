import torch
from torch_geometric.data import Data, Dataset, DataLoader

from typing import Tuple
from torch import Tensor


class FullGraph(Data):
    def __init__(self, fullgraph_input: Tuple[Tensor, Tensor], client_features: Tensor):
        edge_index, edge_weight = fullgraph_input
        super().__init__(x=client_features, edge_index=edge_index, edge_attr=edge_weight)


class ParentGraph(Data):

    def __init__(self, parent_input: Tuple[Tensor, Tensor, Tensor, int], client_features: Tensor):
        client_route_vector, edge_index, edge_weight, num_routes = parent_input
        super().__init__(x=client_features, edge_index=edge_index, edge_attr=edge_weight)
        self.num_routes = num_routes
        self.client_route_vector = client_route_vector
