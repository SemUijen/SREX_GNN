import torch
import torch.nn as nn

from data.utils.DataLoader import get_train_test_loader
from data.utils.ParentGraphDataset import ParentGraphsDataset
from Models import SREXmodel

if __name__ == "__main__":

    label_shape = 50

    dataset = ParentGraphsDataset(root='C:/SREX_GNN/data/test_case', label_shape=50)

    train_dataloader, test_dataloader = get_train_test_loader(dataset, batchsize=3)

    model = SREXmodel(num_node_features=dataset.num_node_features, max_routes_to_swap=50)

    model.train()
    loss_func = nn.MSELoss()

    for count, (p1_data, p2_data, full_graph_data, label) in enumerate(train_dataloader):

        probs = model(p1_data, p2_data)
        print(probs.shape)
        label = torch.tensor(label, dtype=torch.float)
        loss = loss_func(probs, label)
        loss.backward()
        print(loss)
