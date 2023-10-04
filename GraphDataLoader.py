import torch
from torch_geometric.data import Data

from typing import Tuple
from torch import Tensor
from utils.solution_to_model import solutions_to_model, get_example_solutions, get_route_instance
from Models import SREXmodel

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device('cpu')

instance = get_route_instance()
parents = get_example_solutions()

parent1, parent2, client_features = solutions_to_model(instance, parents)


class GraphData(Data):

    def __init__(self, parent_input: Tuple[Tensor, Tensor, Tensor], client_features: Tensor):
        client_route_vector, edge_index, edge_weight, num_routes = parent_input
        super().__init__(x=client_features, edge_index=edge_index, edge_attr=edge_weight)
        self.num_routes = num_routes
        self.client_route_vector = client_route_vector


parent2_data = GraphData(parent_input=parent2, client_features=client_features)
parent1_data = GraphData(parent_input=parent1, client_features=client_features)

model = SREXmodel(num_node_features=6, hidden_dim=16, max_routes_to_swap=50, num_heads=6, dropout=0.2)
model.train()

result = model(parent1_data, parent2_data)

print(result.shape)
