import os

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

from data.utils.DataLoader import get_train_test_loader
from ParentGraphDataset import ParentGraphsDataset
from Models import SREXmodel

if __name__ == "__main__":
    t = torch.tensor([1, 2, 3, 4])
    print(t[[0, 1, 3]])
    dataset = ParentGraphsDataset(root='C:/SREX_GNN/data/test_case')

    train_dataloader, test_dataloader = get_train_test_loader(dataset, batchsize=3)


    model = SREXmodel(num_node_features=dataset.num_node_features)

    model.train()
    loss_func = nn.MSELoss()

    for count, (p1_data, p2_data, full_graph_data, label) in enumerate(train_dataloader):
        print(label[2].shape)
        probs = model(p1_data, p2_data)
        print(probs.shape)
        label = torch.tensor(label, dtype=torch.float)

        loss = loss_func(probs, label)
        loss.backward()
        print(loss)
