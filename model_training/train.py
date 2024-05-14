import torch
from tqdm import tqdm

from data.utils.get_full_graph import get_full_graph
from model_training.utils.Result import Result

from .utils import Metrics, SigmoidVectorizedScaler, Weights, get_lim_labels


def train_model(
    model, device, trainloader, optimizer, loss_func, processed_dir, parameters, epoch
):
    scaler = SigmoidVectorizedScaler(20, device)
    weights = Weights(parameters["weight"])
    metrics = Metrics("Train")
    results = Result(epoch)
    print(f"Training on {len(trainloader)} batches.....")
    # scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.5)
    model.train()
    total_train_loss = 0
    number_of_rows = 0
    with tqdm(total=len(trainloader) - 1) as pbar:
        for count, (p1_data, p2_data, target, instance_idx, acc) in enumerate(
            trainloader
        ):
            p1_data = p1_data.to(device)
            p2_data = p2_data.to(device)

            full_graph, instance_indices = get_full_graph(
                processed_dir, instance_idx, device
            )
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

    # scheduler.step()
    return total_train_loss, (total_train_loss / number_of_rows), metrics, results
