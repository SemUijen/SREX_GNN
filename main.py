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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # TODO: should loss be averaged over each parent combination ?(combinations have different sizes so absolute loss favors smaller instances)
    loss_func = nn.BCELoss(reduction='mean')

    nr_epochs = 1

    for epoch in range(nr_epochs):
        tot_train_loss, avg_train_loss, tot_acc, pos_acc, false_neg, acc_adj = train_model(model, device, train_dataloader, optimizer, loss_func)

        tot_test_loss, avg_test_loss, tot_acc_test, pos_acc_test, false_neg_test, acc_adj_test = test_model(model, device, test_dataloader, loss_func)

        print(
            f'Epoch {epoch + 1} / {nr_epochs} [======] - train_loss(Tot, Avg): {"{:.2f}".format(tot_train_loss)},'
            f' {"{:.2f}".format(avg_train_loss)} - test_loss : {"{:.2f}".format(avg_test_loss)},'
            f' train_scores: {"{:.2f}".format(tot_acc)}, {"{:.2f}".format(pos_acc)}, {"{:.2f}".format(false_neg)}, {"{:.2f}".format(acc_adj)}, '
            f' test_scores {"{:.2f}".format(tot_acc)}, {"{:.2f}".format(pos_acc)}, {"{:.2f}".format(false_neg)}, {"{:.2f}".format(acc_adj_test)}')