
import torch
import torch.nn as nn
import numpy as np
from data.utils.ParentGraphDataset import ParentGraphsDataset
from data.utils.DataLoader import MyDataLoader, MyCollater
from data.utils.BatchSampler import GroupSampler
from train import train_model, test_model
from Models import SREXmodel
from data.utils.get_full_graph import get_full_graph


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    instances_VRP = ["X-n439-k37", "X-n393-k38", "X-n449-k29", "ORTEC-n405-k18", "ORTEC-n510-k23", "X-n573-k30"]
    instances_TW = ["ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35", "R2_8_9", 'R1_4_10']

    instances = instances_VRP + instances_TW

    # Train_test split 772 cvrp files, 386 tw files FILE 88 331 are corrupted
    training = list(range(0, 88)) + list(range(89, 200))
    training = [0, 1,2,3,4,5]
    train_file_names = [f"batch_cvrp_{i}_rawdata.pkl" for i in training]
    # train_file_names.extend([f"batch_tw_{i}_rawdata.pkl" for i in range(308)])

    # test_batches
    test_file_names = [f"batch_cvrp_{i}_rawdata.pkl" for i in range(6, 9)]
    # test_file_names.extend([f"batch_tw_{i}_rawdata.pkl" for i in range(308, 386)])

    trainset = ParentGraphsDataset(root='C:/SREX_GNN/data/model_data', raw_files=train_file_names, instances=instances)
    testset = ParentGraphsDataset(root='C:/SREX_GNN/data/model_data', raw_files=test_file_names, instances=instances)

    sampler = GroupSampler(data_length=len(trainset), group_size=36, batch_size=1)
    train_loader = MyDataLoader(dataset=trainset, batch_sampler=sampler, num_workers=0,
                                collate_fn=MyCollater(None, None))

    sampler = GroupSampler(data_length=len(testset), group_size=36, batch_size=1)
    test_dataloader = MyDataLoader(dataset=testset, batch_sampler=sampler, num_workers=0,
                                collate_fn=MyCollater(None, None))

    model = SREXmodel(num_node_features=trainset.num_node_features)
    model.to(device)

    # TODO: look at optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    # TODO: should loss be averaged over each parent combination ?(combinations have different sizes so absolute loss favors smaller instances)
    loss_func = nn.BCELoss(reduction='mean')

    nr_epochs = 10

    for epoch in range(nr_epochs):
        tot_train_loss, avg_train_loss, tot_acc, pos_acc, false_neg, acc_adj = train_model(model, device,
                                                                                           train_loader, optimizer,
                                                                                           loss_func,
                                                                                           trainset.processed_dir)

        tot_test_loss, avg_test_loss, tot_acc_test, pos_acc_test, false_neg_test, acc_adj_test = test_model(model,
                                                                                                            device,
                                                                                                            test_dataloader,
                                                                                                            loss_func,
                                                                                                            testset.processed_dir)

        print(
            f'Epoch {epoch + 1} / {nr_epochs} [======] - train_loss(Tot, Avg): {"{:.2f}".format(tot_train_loss)},'
            f' {"{:.2f}".format(avg_train_loss)} - test_loss : {"{:.2f}".format(0)},'
            f' train_scores: {"{:.2f}".format(tot_acc)}, {"{:.2f}".format(pos_acc)}, {"{:.2f}".format(false_neg)}, {"{:.2f}".format(acc_adj)},'
            f' test_scores {"{:.2f}".format(tot_acc_test)}, {"{:.2f}".format(pos_acc_test)}, {"{:.2f}".format(false_neg_test)}, {"{:.2f}".format(acc_adj_test)}')


if __name__ == "__main__":


    main()