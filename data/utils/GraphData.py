from torch import Tensor
from torch_geometric.data import Data


class FullGraph(Data):
    def __init__(
        self,
        edge_index: Tensor,
        edge_weight: Tensor,
        client_features: Tensor,
        client_demand: Tensor,
        client_pos: Tensor,
        depot_pos: Tensor,
        client_time: Tensor,
    ):
        super().__init__(
            x=client_features,
            edge_index=edge_index,
            edge_attr=edge_weight,
            pos=client_pos,
        )
        self.client_demand = client_demand
        self.depot_pos = depot_pos
        self.client_time = client_time


class ParentGraph(Data):
    def __init__(
        self,
        client_route_vector: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        num_routes: int,
        client_features: Tensor,
        client_demand: Tensor,
        client_pos: Tensor,
        depot_pos: Tensor,
        client_time: Tensor,
    ):
        super().__init__(
            x=client_features,
            edge_index=edge_index,
            edge_attr=edge_weight,
            pos=client_pos,
        )
        self.num_routes = num_routes
        self.client_route_vector = client_route_vector
        self.client_demand = client_demand
        self.depot_pos = depot_pos
        self.client_time = client_time
