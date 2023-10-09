import os.path as osp
import pickle
from typing import Union, List, Tuple
import numpy as np

import torch
from torch_geometric.data import Dataset
from data.utils.GraphData import FullGraph, ParentGraph
from data.utils import SolutionTransformer


class ParentGraphsDataset(Dataset):
    def __init__(self, root: str, label_shape: int, is_processed: bool = False,
                 pre_transform: SolutionTransformer = SolutionTransformer(),
                 transform=None, pre_filter=None):
        self.processed_files = []
        self.is_processed = is_processed
        self.parent_couple_idx = []
        self.instance_idx = []
        self.label_shape = label_shape
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> List[str]:
        return ["X-n439-k37_rawdata.pkl"]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self.processed_files

    @staticmethod
    def read_pickle(raw_path):
        with open(raw_path, "rb") as file:
            raw_data = pickle.load(file)
        return raw_data

    def transform_labels(self, labels):
        # TODO: look at padding options
        labels_transformed = []
        for label in labels:
            x, y, z = label.shape
            transformed_label = np.pad(label, pad_width=(
            (0, self.label_shape - x), (0, self.label_shape - y), (0, self.label_shape - z - 1)))

            labels_transformed.append(transformed_label)

        return labels_transformed

    def process(self) -> None:
        if not self.is_processed:
            idx = 0
            for raw_path in self.raw_paths:
                # Read data from `raw_path`.
                raw_data = self.read_pickle(raw_path)

                # process_whole_graph
                route_instance = raw_data['route_instance_name']
                file_name = f'FullGraph_{idx}.pt'
                if self.pre_transform:
                    edge_index, edge_weight, client_features = self.pre_transform(instance_name=route_instance,
                                                                                  get_full_graph=True)

                    data = FullGraph(edge_index, edge_weight, client_features)
                    torch.save(data, osp.join(self.processed_dir, file_name))

                # for getter function #TODO: fix something better works for now
                InstanceIdx = [idx] * len(raw_data["parent_couple_idx"])
                self.instance_idx.extend(InstanceIdx)

                # add couple labels
                self.parent_couple_idx.extend(raw_data['parent_couple_idx'])

                # save labels:
                labels = self.transform_labels(raw_data["labels"])
                file_name = f'labels_{idx}.pt'
                torch.save(labels, osp.join(self.processed_dir, file_name))

                idx += 1
                for solution in raw_data["parent_routes"]:

                    if self.pre_transform:
                        client_route_vector, edge_index, edge_weight, num_routes = self.pre_transform(
                            instance_name=route_instance, get_full_graph=False,
                            parent_solution=solution)

                        data = ParentGraph(client_route_vector, edge_index, edge_weight, num_routes, client_features)
                        file_name = f'ParentGraphs_{idx}.pt'
                        self.processed_files.append(file_name)
                        torch.save(data, osp.join(self.processed_dir, file_name))
                        idx += 1

                    else:
                        raise "No pre_transform"

    def len(self) -> int:
        return len(self.parent_couple_idx)

    def get(self, idx):
        p1_idx, p2_idx = self.parent_couple_idx[idx]
        instance_idx = self.instance_idx[idx]

        p1_data = torch.load(osp.join(self.processed_dir, f'ParentGraphs_{p1_idx}.pt'))
        p2_data = torch.load(osp.join(self.processed_dir, f'ParentGraphs_{p2_idx}.pt'))

        full_graph_data = torch.load(osp.join(self.processed_dir, f'FullGraph_{instance_idx}.pt'))
        label = torch.load(osp.join(self.processed_dir, f'labels_{instance_idx}.pt'))

        return p1_data, p2_data, full_graph_data, torch.tensor(label[idx])
