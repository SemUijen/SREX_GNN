import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.nn.functional import softmax, sigmoid
from utils.metrics import Metrics
from utils.LabelScalers import SigmoidVectorizedScaler
from tqdm import tqdm
from data.utils.get_full_graph import get_full_graph
from Result import Result
import os.path as osp


def train_model(model, device, trainloader, optimizer, loss_func, processed_dir, parameters, epoch):
    scaler = SigmoidVectorizedScaler(20, device)
    weights = Weights(parameters['weight'])
    metrics = Metrics("Train")
    results = Result(epoch)
    print(f'Training on {len(trainloader)} batches.....')
    #scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.5)
    model.train()
    total_train_loss = 0
    number_of_rows = 0
    with tqdm(total=len(trainloader) - 1) as pbar:

        for count, (p1_data, p2_data, target, instance_idx, acc) in enumerate(trainloader):
            p1_data = p1_data.to(device)
            p2_data = p2_data.to(device)

            full_graph, instance_indices = get_full_graph(processed_dir, instance_idx, device)
            full_graph = full_graph.to(device)

            output, batch = model(p1_data, p2_data, full_graph, instance_indices, epoch)

            optimizer.zero_grad()
            loss = torch.tensor(0.0, device=device)
            for i in range(len(p1_data)):
                label = torch.tensor(target[i].label, device=device, dtype=torch.float)

                p1, p2 = p1_data.num_routes[i], p2_data.num_routes[i]
                shape = (min(p1, p2), p1, p2)
                label = get_lim_labels(label, shape)

                loss_func.weight = weights(label, output[batch == i], acc[i], device)
                if parameters["binary_label"]:
                    label = torch.where(label > 0, 1.0, 0.0)
                else:
                    label = scaler(label)
                    label = torch.sigmoid(label)


                results.add(label, output[batch == i], shape, instance_idx[i])

                loss1 = loss_func(output[batch == i], label)
                loss += loss1
                metrics(output[batch == i], label)
                number_of_rows += 1

            total_train_loss += loss.detach()
            loss.backward()
            optimizer.step()
            pbar.update()

    #scheduler.step()
    return total_train_loss, (total_train_loss / number_of_rows), metrics, results


def test_model(model, device, testloader, loss_func, processed_dir, parameters, epoch):
    scaler = SigmoidVectorizedScaler(20, device)
    weights = Weights(parameters['weight'])
    metrics = Metrics("test")
    model.eval()
    loss = torch.tensor(0.0, device=device)
    number_of_rows = 0
    with torch.no_grad():
        for count, (p1_data, p2_data, target, instance_idx, acc) in enumerate(testloader):
            p1_data = p1_data.to(device)
            p2_data = p2_data.to(device)

            full_graph, instance_indices = get_full_graph(processed_dir, instance_idx, device)
            full_graph = full_graph.to(device)
            output, batch = model(p1_data, p2_data, full_graph, instance_indices, epoch)

            for i in range(len(p1_data)):
                label = torch.tensor(target[i].label, device=device, dtype=torch.float)
                p1, p2 = p1_data.num_routes[i], p2_data.num_routes[i]
                shape = (min(p1, p2), p1, p2)
                label = get_lim_labels(label, shape)
                loss_func.weight = weights(label, output[batch == i], acc[i], device)
                if parameters["binary_label"]:
                    label = torch.where(label > 0, 1.0, 0.0)

                else:
                    label = scaler(label)
                    label = torch.sigmoid(label)

                loss1 = loss_func(output[batch == i], label)
                loss += loss1
                metrics(output[batch == i], label)
                number_of_rows += 1

        return loss, (loss / number_of_rows), metrics

def get_lim_labels(label, shape):
    max_move, p1, p2 = shape
    list_test = []
    copy = label.clone()
    copy = copy.reshape(shape)
    for i in range(max_move):
        for i2 in range(p1):
            if i2 < p2:
                list_test.append([i, i2, i2])
            else:
                list_test.append([i, i2, 0])

    indices = torch.tensor([list_test])
    result = copy[indices[:, :, 0], indices[:, :, 1], indices[:, :, 2]]
    return result.flatten()

class Weights:

    def __init__(self, weight_type: bool):
        self.weight_type = weight_type

    def __call__(self, label, prediction, acc_score, device):

        if self.weight_type is None:
            return torch.ones(label.shape, device=device)

        if self.weight_type == "strong":
            return self.get_weight_strong(label, acc_score, device)

        if self.weight_type == "confuse":
            return self.conf_matrix_weight(label, prediction, device)

    def get_weight_strong(self, label, acc_score, device):
        tot = len(label)
        pos = round((acc_score * len(label)).item())
        neg = tot - pos

        if pos == 0 or neg == 0:
            return torch.ones(label.shape, device=device)

        pos_w = tot / (2 * pos)
        neg_w = tot / (2 * neg)

        weights = torch.where(label > 0, pos_w, neg_w).to(device)
        return weights

    def conf_matrix_weight(self, label, prediction, device):
        binary_predict = torch.where(prediction > 0.5, 1, 0)
        binary_label = torch.where(label > 0, 1, 0)
        equality = torch.eq(binary_predict, binary_label)

        weight = torch.ones(label.shape, device=device)
        pos_pred = equality[binary_predict.nonzero()]
        if len(pos_pred) == 0:
            pos_acc = 1
        else:
            weight[torch.where(pos_pred == False)[0]] = 1.2

        weight[torch.where(label > 0)] = 5
        return weight
