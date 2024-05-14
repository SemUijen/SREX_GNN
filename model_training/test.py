import torch
from torch import Tensor

from data.utils.get_full_graph import get_full_graph

from .utils import Metrics, SigmoidVectorizedScaler, Weights, get_lim_labels


def test_model(model, device, testloader, loss_func, processed_dir, parameters, epoch):
    scaler = SigmoidVectorizedScaler(20, device)
    weights = Weights(parameters["weight"])
    metrics = Metrics("test")
    model.eval()
    loss = torch.tensor(0.0, device=device)
    number_of_rows = 0
    with torch.no_grad():
        for count, (p1_data, p2_data, target, instance_idx, acc) in enumerate(
            testloader
        ):
            p1_data = p1_data.to(device)
            p2_data = p2_data.to(device)

            full_graph, instance_indices = get_full_graph(
                processed_dir, instance_idx, device
            )
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
