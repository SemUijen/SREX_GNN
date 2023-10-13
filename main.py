import torch
import torch.nn as nn
import numpy as np
from data.utils.DataLoader import get_train_test_loader
from data.utils.ParentGraphDataset import ParentGraphsDataset

from train import train_model, test_model
from Models import SREXmodel

if __name__ == "__main__":

    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

    label_shape = 40
    dataset = ParentGraphsDataset(root='C:/SREX_GNN/data/test_case', label_shape=label_shape)

    # TODO: hoe het beste batches: Full graph * 8 verschillende groepen + (4*4 parents combinations)
    train_dataloader, test_dataloader = get_train_test_loader(dataset, batchsize=3)

    model = SREXmodel(num_node_features=dataset.num_node_features, max_routes_to_swap=label_shape)
    model.to(device)

    # TODO: look at optimizers
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    loss_func = nn.CrossEntropyLoss()


    nr_epochs = 5
    for epoch in range(nr_epochs):
        train_loss = train_model(model, device, train_dataloader, optimizer, loss_func)

        #test_pred, test_label = test_model(model, device, test_dataloader)

        #test_loss = loss_func(test_pred.flatten(), test_label.flatten())
        #print(
        #   f'Epoch {epoch + 1} / {nr_epochs} [==============================] - train_loss : {train_loss} - test_loss : {test_loss}')

