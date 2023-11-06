import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.nn.functional import softmax, sigmoid
from utils.metrics import Metrics
from utils.LabelScalers import SigmoidVectorizedScaler
from tqdm import tqdm
from data.utils.get_full_graph import get_full_graph
from Result import Result


def train_model(model, device, trainloader, optimizer, loss_func, processed_dir, parameters, epoch):
    scaler = SigmoidVectorizedScaler(20, device)
    weights = Weights(parameters['weight'])
    metrics = Metrics("Train")
    results = Result(epoch)
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
            loss = torch.tensor(0.0)
            temp_lab = torch.tensor([])
            temp_weight = torch.tensor([])
            for i in range(len(p1_data)):
                label = torch.tensor(target[i].label, device=device, dtype=torch.float)
                temp_weight = torch.cat((temp_weight, weights(label, output[batch == i], acc[i], device)))

                if parameters["binary_label"]:
                    label = torch.where(label > 0, 1.0, 0.0)
                else:
                    label = scaler(label)
                    label = torch.sigmoid(label)

                p1, p2 = p1_data.num_routes[i], p2_data.num_routes[i]
                shape = (min(p1, p2), p1, p2)

                if i % 12 == 0:
                    results.add(label, output[batch == i], shape, instance_idx[i])

                #loss1 = loss_func(output[batch == i], label)
                #loss += loss1
                metrics(output[batch == i], label)
                number_of_rows += 1
                temp_lab = torch.cat((temp_lab, label))


            loss_func.weight = temp_weight
            loss = loss_func(output[:temp_lab.shape[0]], temp_lab)

            total_train_loss += loss
            loss.backward()
            optimizer.step()
            pbar.update()

    return total_train_loss, (total_train_loss / number_of_rows), metrics, results


def test_model(model, device, testloader, loss_func, processed_dir, parameters):
    scaler = SigmoidVectorizedScaler(20, device)
    weights = Weights(parameters['weight'])
    metrics = Metrics("test")
    model.eval()
    loss = torch.tensor(0.0)
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

        weight[torch.where(label > 0)] = 2
        return weight
