import os.path as osp
import pickle
from typing import Union, List, Tuple
import numpy as np

import torch
from torch_geometric.data import Dataset
from data.utils.GraphData import FullGraph, ParentGraph
from data.utils import SolutionTransformer


class MyLabel:
    def __init__(self, label):
        self.label = label


class ParentGraphsDataset(Dataset):
    def __init__(self, root: str, raw_files: List[str], instances: List[str], is_processed: bool = False,
                 pre_transform: SolutionTransformer = SolutionTransformer(),
                 transform=None, pre_filter=None):
        self.processed_files = []
        self.is_processed = is_processed
        self.parent_couple_idx = []
        self.instance_idx = []
        self.labels = []
        self.accuracy = []
        self.accuracy_limit = []
        self.raw_files = raw_files
        self.instances = instances
        self.instance_dict = {}
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> List[str]:
        return self.raw_files

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self.processed_files

    @staticmethod
    def read_pickle(raw_path):
        with open(raw_path, "rb") as file:
            raw_data = pickle.load(file)
        return raw_data

    def process(self) -> None:
        if not self.is_processed:
            # First get FullGraphs
            idx = 0
            for instance in self.instances:
                edge_index, edge_weight, client_features = self.pre_transform(instance_name=instance,
                                                                              get_full_graph=True)
                data = FullGraph(edge_index, edge_weight, client_features)
                file_name = f'FullGraph_{idx}.pt'
                torch.save(data, osp.join(self.processed_dir, file_name))
                self.instance_dict[instance] = idx
                idx += 1

            for raw_path in self.raw_paths:
                # Read data from `raw_path`.
                raw_data = self.read_pickle(raw_path)

                for batch in range(len(raw_data["parent_routes"])):
                    # process_whole_graph
                    route_instance = raw_data["instances"][batch]

                    # for getting the correct Full_graph
                    InstanceIdx = [self.instance_dict[route_instance]] * 12
                    self.instance_idx.extend(InstanceIdx)

                    # add couple labels
                    self.parent_couple_idx.extend(raw_data['parent_couple_idx'][batch])
                    self.labels.extend(raw_data["labels"][batch])
                    self.accuracy.extend(raw_data["random_acc"][batch])
                    self.accuracy_limit.extend(raw_data["random_acc_limit"][batch])

                    for i in range(len(raw_data["parent_routes"][batch])):
                        solution = raw_data["parent_routes"][batch][i]
                        idx = raw_data["parent_ids"][batch][i]

                        client_route_vector, edge_index, edge_weight, num_routes, client_features = self.pre_transform(
                            instance_name=route_instance, get_full_graph=False,
                            parent_solution=solution)

                        data = ParentGraph(client_route_vector, edge_index, edge_weight, num_routes,
                                           client_features)
                        file_name = f'ParentGraph_{idx}.pt'
                        self.processed_files.append(file_name)
                        torch.save(data, osp.join(self.processed_dir, file_name))

    def len(self) -> int:
        return len(self.parent_couple_idx)

    def get(self, idx):
        p1_idx, p2_idx = self.parent_couple_idx[idx]
        instance_idx = self.instance_idx[idx]

        p1_data = torch.load(osp.join(self.processed_dir, f'ParentGraph_{p1_idx}.pt'))
        p2_data = torch.load(osp.join(self.processed_dir, f'ParentGraph_{p2_idx}.pt'))

        full_graph = torch.load(osp.join(self.processed_dir, f'FullGraph_{instance_idx}.pt'))

        label = self.labels[idx]
        # because of varying sizes of labels. The labels are put in dict so they can be stacked by dataloader

        label = MyLabel(label)
        return p1_data, p2_data, full_graph, label

    def get_accuracy_scores(self) -> Tuple[float, float]:
        limit_acc = sum(self.accuracy_limit) / len(self)
        random_acc = sum(self.accuracy) / len(self)
        return limit_acc, random_acc
