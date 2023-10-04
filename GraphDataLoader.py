import os
import pickle
import pandas as pd

import torch

from torch_geometric.data import Data, DataLoader
from Route_to_input import get_edge_features_from_instance, get_adj_matrix_from_solutions, \
    get_node_features_from_instance, get_example_solutions, get_route_instance
from Models import SREXmodel

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device('cpu')

node_features_m = get_node_features_from_instance()
edge_features_m = get_edge_features_from_instance()
edge_m1, edge_m2 = get_adj_matrix_from_solutions(get_example_solutions())
num_routes = get_example_solutions()[0].num_routes()

edge_index = edge_m1.nonzero().t()
row, col = edge_index
edge_weight = edge_features_m[row, col]


data = Data(x=node_features_m, edge_index=edge_index, edge_attr=edge_weight)
data.num_routes = num_routes
print(data.is_directed())
print(data)

model = SREXmodel(num_node_features=6, hidden_dim=16, max_routes_to_swap=50, num_heads=6, dropout=0.2)
model.train()

result = model(data, data)

print(result.shape)
