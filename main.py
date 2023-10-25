import torch
import torch.nn as nn
import numpy as np
from data.utils.ParentGraphDataset import ParentGraphsDataset
from data.utils.DataLoader import MyDataLoader, MyCollater
from data.utils.BatchSampler import GroupSampler
from train import train_model, test_model
from Models import SREXmodel

if __name__ == "__main__":

    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

    instances_VRP = ["X-n439-k37", "X-n393-k38", "X-n449-k29", "ORTEC-n405-k18", "ORTEC-n510-k23", "X-n573-k30"]
    instances_TW = ["ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35", "R2_8_9", 'R1_4_10']

    instances = instances_VRP + instances_TW

    raw_file_names = [f"batch_cvrp_{i}_rawdata.pkl" for i in range(2)]
    raw_file_names.extend([f"batch_tw_{i}_rawdata.pkl" for i in range(2)])

    dataset = ParentGraphsDataset(root='C:/SREX_GNN/data/model_data', raw_files=raw_file_names, instances=instances)


    sampler = GroupSampler(data_length=len(dataset), group_size=12, batch_size=3)
    train_loader = MyDataLoader(dataset=dataset, batch_sampler=sampler, num_workers=0,
                                collate_fn=MyCollater(None, None))

    model = SREXmodel(num_node_features=dataset.num_node_features)
    model.to(device)

    # TODO: look at optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # TODO: should loss be averaged over each parent combination ?(combinations have different sizes so absolute loss favors smaller instances)
    loss_func = nn.BCELoss(reduction='mean')

    nr_epochs = 1

    for epoch in range(nr_epochs):
        tot_train_loss, avg_train_loss, tot_acc, pos_acc, false_neg, acc_adj = train_model(model, device,
                                                                                           train_loader, optimizer,
                                                                                           loss_func)

        # tot_test_loss, avg_test_loss, tot_acc_test, pos_acc_test, false_neg_test, acc_adj_test = test_model(model,
        #                                                                                                     device,
        #                                                                                                     test_dataloader,
        #                                                                                                     loss_func)

        print(
            f'Epoch {epoch + 1} / {nr_epochs} [======] - train_loss(Tot, Avg): {"{:.2f}".format(tot_train_loss)},'
            f' {"{:.2f}".format(avg_train_loss)} - test_loss : {"{:.2f}".format(0)},'
            f' train_scores: {"{:.2f}".format(tot_acc)}, {"{:.2f}".format(pos_acc)}, {"{:.2f}".format(false_neg)}, {"{:.2f}".format(acc_adj)},')
            #f' test_scores {"{:.2f}".format(tot_acc_test)}, {"{:.2f}".format(pos_acc_test)}, {"{:.2f}".format(false_neg_test)}, {"{:.2f}".format(acc_adj_test)}')
