import torch
import torch.nn as nn
import numpy as np
from data.utils.ParentGraphDataset import ParentGraphsDataset
from data.utils.DataLoader import MyDataLoader, MyCollater
from data.utils.BatchSampler import GroupSampler
from train import train_model, test_model
from Models import SREXmodel
from data.utils.get_full_graph import get_full_graph
import os
import os.path as osp

def main(parameters):
    print(os.getcwd())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    instances_VRP = ["X-n439-k37", "X-n393-k38", "X-n449-k29", "ORTEC-n405-k18", "ORTEC-n510-k23", "X-n573-k30"]
    instances_TW = ["ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35", "R2_8_9", 'R1_4_10']

    instances = instances_VRP + instances_TW

    # Train_test split 772 cvrp files, 386 tw files FILE 88 331 are corrupted
    training = list(range(0, 88)) + list(range(89, 331)) + list(range(332, 618))
    test = list(range(618, 772))
    train_file_names = [f"batch_cvrp_{i}_rawdata.pkl" for i in training]
    # train_file_names.extend([f"batch_tw_{i}_rawdata.pkl" for i in range(308)])

    # test_batches
    test_file_names = [f"batch_cvrp_{i}_rawdata.pkl" for i in test]
    # test_file_names.extend([f"batch_tw_{i}_rawdata.pkl" for i in range(308, 386)])


    trainset = ParentGraphsDataset(root=osp.join(os.getcwd(), 'data/model_data'), raw_files=train_file_names, instances=instances)
    testset = ParentGraphsDataset(root=osp.join(os.getcwd(), 'data/model_data'), raw_files=test_file_names, instances=instances)

    sampler = GroupSampler(data_length=len(trainset), group_size=36, batch_size=1)
    train_loader = MyDataLoader(dataset=trainset, batch_sampler=sampler, num_workers=0,
                                collate_fn=MyCollater(None, None))

    sampler = GroupSampler(data_length=len(testset), group_size=36, batch_size=1)
    test_loader = MyDataLoader(dataset=testset, batch_sampler=sampler, num_workers=0,
                                   collate_fn=MyCollater(None, None))

    model = SREXmodel(num_node_features=trainset.num_node_features, hidden_dim=parameters["hidden_dim"], num_heads=parameters['num_heads'])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"])

    loss_func = nn.BCELoss(reduction='mean')

    nr_epochs = parameters["epochs"]
    print(trainset.get_accuracy_scores())
    print(testset.get_accuracy_scores())
    f1_best = 0
    select_acc = 0
    for epoch in range(nr_epochs):
        tot_train_loss, avg_train_loss, train_metric = train_model(model, device,
                                                                   train_loader, optimizer,
                                                                   loss_func,
                                                                   trainset.processed_dir,
                                                                   parameters)

        tot_test_loss, avg_test_loss, test_metric = test_model(model,
                                                               device,
                                                               test_loader,
                                                               loss_func,
                                                               testset.processed_dir,
                                                               parameters)

        print(
            f'Epoch {epoch + 1} / {nr_epochs} [======] - train_loss(Tot, Avg): {"{:.2f}".format(tot_train_loss)},'
            f' {"{:.2f}".format(avg_train_loss)} - test_loss : {"{:.2f}".format(avg_test_loss)}, \n'
            f"{train_metric} \n"
            f"{test_metric}")

        if train_metric.f1 > f1_best or train_metric.select_acc > select_acc:

            obj_dict = {"model_state": model.state_dict(),
                        "Metrics_train": train_metric,
                        "Metrics_test": test_metric,
                        }
            torch.save(obj_dict,
                       f'{trainset.root}/model_states/SrexGNN_{parameters["run"]}_{epoch}_{"{:.2f}".format(train_metric.f1)}_{"{:.2f}".format(train_metric.select_acc)}')

            if train_metric.f1 > f1_best:
                f1_best = train_metric.f1
            if train_metric.select_acc > select_acc:
                select_acc = train_metric.select_acc


if __name__ == "__main__":
    # TODO create parameters loop
    # use different weights https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/#h-what-are-class-weights
    # use different learning rate
    # check if more features are needed
    # Check what is happening with sigmoid output
    # TODO average summed routes

    #Training Parameters
    parameters = {"learning_rate": 0.001,
                  "pos_weight": 5,
                  "epochs": 2,
                  "binary_label": True,
                  "run": 2,
                  "hidden_dim": 64,
                  "num_heads": 8,
    }

    main(parameters)







