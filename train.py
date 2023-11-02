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
    metrics = Metrics("Train")
    print(f'Training on {len(trainloader)} batches.....')
    model.train()
    total_train_loss = 0
    number_of_rows = 0
    with tqdm(total=len(trainloader) - 1) as pbar:

        for count, (p1_data, p2_data, target, instance_idx, acc) in enumerate(trainloader):
            p1_data = p1_data.to(device)
            p2_data = p2_data.to(device)
            full_graph, instance_indices = get_full_graph(processed_dir, instance_idx, device)
            full_graph = full_graph.to(device)
            optimizer.zero_grad()
            output, batch = model(p1_data, p2_data, full_graph, instance_indices)
            loss = 0

            for i in range(len(p1_data)):
                label = torch.tensor(target[i].label, device=device, dtype=torch.float)
                loss_func.weight = get_weight(label, acc[i])

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
    metrics = Metrics("test")
    model.eval()
    loss = 0
    number_of_rows = 0
    with torch.no_grad():
        for count, (p1_data, p2_data, target, instance_idx, acc) in enumerate(testloader):
            p1_data = p1_data.to(device)
            p2_data = p2_data.to(device)
            full_graph, instance_indices = get_full_graph(processed_dir, instance_idx, device)
            full_graph = full_graph.to(device)
            output, batch = model(p1_data, p2_data, full_graph, instance_indices)

            for i in range(len(p1_data)):
                label = torch.tensor(target[i].label, device=device, dtype=torch.float)
                loss_func.weight = get_weight(label, acc[i])

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

def get_weight(label, acc_score):
    tot = len(label)
    pos = round((acc_score*len(label)).item())
    neg = tot - pos

    if pos==0 or neg ==0:
        return torch.ones(label.shape)

    pos_w = tot/(2*pos)
    neg_w = tot/(2*neg)

    return torch.where(label > 0, pos_w, neg_w)