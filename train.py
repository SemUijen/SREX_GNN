import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch import softmax


def train_model(model, device, trainloader, optimizer, loss_func):
    print(f'Training on {len(trainloader)} samples.....')
    model.train()

    for count, (p1_data, p2_data, full_graph_data, target) in enumerate(trainloader):
        p1_data = p1_data.to(device)
        p2_data = p2_data.to(device)
        optimizer.zero_grad()
        output, batch = model(p1_data, p2_data)
        loss = 0
        # TODO: Average losses?
        for i in range(len(p1_data)):
            label = torch.tensor(target[i].label)
            soft_max_label = softmax(label, 0, torch.float)
            loss1 = loss_func(output[batch == i], soft_max_label)
            loss += loss1

        loss.backward()
        optimizer.step()

    # TODO: create accuracy functions: Absolute vs Current SREX
    return loss


def test_model(model, device, testloader, loss_func):
    model.eval()
    loss = 0
    labels = torch.Tensor()
    with torch.no_grad():
        for count, (p1_data, p2_data, full_graph_data, target) in enumerate(testloader):
            p1_data = p1_data.to(device)
            p2_data = p2_data.to(device)
            output, batch = model(p1_data, p2_data)

            for i in range(len(p1_data)):
                label = torch.tensor(target[i].label)
                soft_max_label = softmax(label, 0, torch.float)
                loss1 = loss_func(output[batch==i], soft_max_label)
                loss += loss1

        return loss
