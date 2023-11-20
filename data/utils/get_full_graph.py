import os.path as osp
from typing import Tuple, List

from torch_geometric.data import Batch
from torch import Tensor
import torch


def get_full_graph(processed_dir: str, instance_idx: List[int], device) -> Tuple[Batch, Tensor]:
    full_graph_data = []
    instances = torch.unique(instance_idx)

    id_full = 0
    instance_batch = torch.tensor([], device=device)
    for idx in instances:
        repeated = len(instance_idx[instance_idx == idx])
        graph = torch.load(osp.join(processed_dir, f'FullGraph_{idx}.pt'))
        full_graph_data.append(graph)
        instance_batch = torch.cat(
            (instance_batch, torch.tensor(id_full, device=device).repeat(repeated * len(graph.x))))
        id_full += 1
    full_graph = Batch.from_data_list(full_graph_data)

    return full_graph, instance_batch
