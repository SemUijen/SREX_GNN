import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch import softmax


def train_model(model, device, trainloader, optimizer, loss_func):
    print(f'Training on {len(trainloader)} samples.....')
    model.train()
    scheduler = MultiStepLR(optimizer, milestones=[1, 5], gamma=0.5)
    for count, (p1_data, p2_data, full_graph_data, label) in enumerate(trainloader):
        p1_data = p1_data.to(device)
        p2_data = p2_data.to(device)
        optimizer.zero_grad()
        output = model(p1_data, p2_data)
        soft_max_label = softmax(label, -1, torch.float)
        loss = loss_func(output, soft_max_label)
        loss.backward()
        optimizer.step()
    scheduler.step()
    # TODO: create accuracy functions: Absolute vs Current SREX
    return loss


def test_model(model, device, testloader):
    model.eval()
    predictions = torch.Tensor()
    labels = torch.Tensor()
    with torch.no_grad():
        for count, (p1_data, p2_data, full_graph_data, label) in enumerate(testloader):
            p1_data = p1_data.to(device)
            p2_data = p2_data.to(device)
            output = model(p1_data, p2_data)
            # TODO: Test if softmax is done correctly
            soft_max_label = softmax(label, -1, torch.float)

            predictions = torch.cat((predictions, output))
            labels = torch.cat((labels, soft_max_label))

        return predictions, labels