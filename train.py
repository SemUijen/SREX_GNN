import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.functional import softmax, sigmoid
from utils.metrics import Metrics
from utils.LabelScalers import SigmoidVectorizedScaler
from tqdm import tqdm
from data.utils.get_full_graph import get_full_graph


def train_model(model, device, trainloader, optimizer, loss_func, processed_dir, parameters):
    scaler = SigmoidVectorizedScaler(20, device)
    weights = Weights(parameters['strong'])
    metrics = Metrics("Train")
    print(f'Training on {len(trainloader)} batches.....')
    model.train()
    total_train_loss = 0
    number_of_rows = 0
    with tqdm(total=len(trainloader) - 1) as pbar:

        for count, (p1_data, p2_data, target, instance_idx, acc) in enumerate(trainloader):
            p1_data = p1_data.to(device)
            p2_data = p2_data.to(device)
            if parameters["fullgraph"]:
                full_graph, instance_indices = get_full_graph(processed_dir, instance_idx, device)
                full_graph = full_graph.to(device)
                output, batch = model(p1_data, p2_data, full_graph, instance_indices)
            else:
                output, batch = model(p1_data, p2_data)
            optimizer.zero_grad()
            loss = 0

            for i in range(len(p1_data)):
                label = torch.tensor(target[i].label, device=device, dtype=torch.float)
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

            total_train_loss += loss
            loss.backward()
            optimizer.step()
            pbar.update()

    return total_train_loss, (total_train_loss / number_of_rows), metrics


def test_model(model, device, testloader, loss_func, processed_dir, parameters):
    scaler = SigmoidVectorizedScaler(20, device)
    weights = Weights(parameters['strong'])
    metrics = Metrics("test")
    model.eval()
    loss = 0
    number_of_rows = 0
    with torch.no_grad():
        for count, (p1_data, p2_data, target, instance_idx, acc) in enumerate(testloader):
            p1_data = p1_data.to(device)
            p2_data = p2_data.to(device)

            if parameters["fullgraph"]:
                full_graph, instance_indices = get_full_graph(processed_dir, instance_idx, device)
                full_graph = full_graph.to(device)
                output, batch = model(p1_data, p2_data, full_graph, instance_indices)
            else:
                output, batch = model(p1_data, p2_data)

            for i in range(len(p1_data)):
                label = torch.tensor(target[i].label, device=device, dtype=torch.float)
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


class Weights:

    def __init__(self, strong: bool):
        self.strong = strong

    def __call__(self, label, prediction, acc_score, device):

        if self.strong:
            return self.get_weight_strong(label, acc_score, device)
        else:
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
            weight[torch.where(pos_pred == False)[0]] = 1.5

        weight[torch.where(label > 0)] = 2
        return weight
