import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.functional import softmax, sigmoid


def train_model(model, device, trainloader, optimizer, loss_func):
    print(f'Training on {len(trainloader)} samples.....')
    model.train()
    total_train_loss = 0
    number_of_rows = 0
    for count, (p1_data, p2_data, full_graph_data, target) in enumerate(trainloader):
        p1_data = p1_data.to(device)
        p2_data = p2_data.to(device)
        optimizer.zero_grad()
        output, batch = model(p1_data, p2_data)
        loss = 0
        # TODO: Average losses?
        for i in range(len(p1_data)):
            label = torch.tensor(target[i].label, device='cuda', dtype=torch.float)
            soft_max_label = torch.sigmoid(label)
            loss1 = loss_func(output[batch == i], soft_max_label)
            loss += loss1
            number_of_rows += 1
        total_train_loss += loss
        loss.backward()
        optimizer.step()

    # TODO: create accuracy functions: Absolute vs Current SREX
    return total_train_loss/number_of_rows


def test_model(model, device, testloader, loss_func):
    model.eval()
    loss = 0
    number_rows = 0
    with torch.no_grad():
        for count, (p1_data, p2_data, full_graph_data, target) in enumerate(testloader):
            p1_data = p1_data.to(device)
            p2_data = p2_data.to(device)
            output, batch = model(p1_data, p2_data)

            for i in range(len(p1_data)):
                label = torch.tensor(target[i].label, device='cuda', dtype=torch.float)
                soft_max_label = torch.sigmoid(label)
                loss1 = loss_func(output[batch == i], soft_max_label)
                loss += loss1
                number_rows += 1

        return loss / number_rows
