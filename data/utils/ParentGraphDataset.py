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
    def __init__(self, root: str, label_shape: int, is_processed: bool = False,
                 pre_transform: SolutionTransformer = SolutionTransformer(),
                 transform=None, pre_filter=None):
        self.processed_files = []
        self.is_processed = is_processed
        self.parent_couple_idx = []
        self.instance_idx = []
        self.label_shape = label_shape
        self.labels = []
        self.accuracy = []
        self.accuracy_limit = []
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> List[str]:
        return ["batch_12_rawdata.pkl", "batch_22_rawdata.pkl", "batch_32_rawdata.pkl", "batch_42_rawdata.pkl", "batch_52_rawdata.pkl","batch_62_rawdata.pkl","batch_72_rawdata.pkl","batch_82_rawdata.pkl","batch_112_rawdata.pkl","batch_312_rawdata.pkl","batch_412_rawdata.pkl","batch_512_rawdata.pkl","batch_712_rawdata.pkl","batch_812_rawdata.pkl"]
    @property
    def instance_names(self) -> List[str]:
        return ["X-n439-k37", "X-n393-k38", "X-n449-k29", "ORTEC-n405-k18", "ORTEC-n510-k23", "X-n573-k30", "ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35", "R2_8_9", "X-n439-k37", "X-n449-k29", "ORTEC-n405-k18", "ORTEC-n510-k23", "ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35", "R2_8_9"]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self.processed_files

    @staticmethod
    def read_pickle(raw_path):
        with open(raw_path, "rb") as file:
            raw_data = pickle.load(file)
        return raw_data

    def transform_labels(self, labels):
        # TODO: look at padding options (current is zeropadding)
        labels_transformed = []
        for label in labels:
            x, y, z = label.shape
            transformed_label = np.pad(label, pad_width=(
                (0, self.label_shape - x), (0, self.label_shape - y), (0, self.label_shape - z - 1)))

            labels_transformed.append(transformed_label)

        return labels_transformed

    def flatten_labels(self, labels):
        # TODO: look at padding options (current is zeropadding)
        labels_transformed = []
        for label in labels:
            flattened_label = label.flatten()
            labels_transformed.append(list(flattened_label))

        return labels_transformed

    def process(self) -> None:
        if not self.is_processed:
            idx = 1
            idx2 = 0
            for raw_path in self.raw_paths:
                # Read data from `raw_path`.
                raw_data = self.read_pickle(raw_path)

                for batch in range(len(raw_data["parent_routes"])):

                    # process_whole_graph
                    route_instance = self.instance_names[idx2]
                    file_name = f'FullGraph_{idx2}.pt'
                    idx2 += 1
                    if self.pre_transform:
                        edge_index, edge_weight, client_features = self.pre_transform(instance_name=route_instance,
                                                                                      get_full_graph=True)

                        data = FullGraph(edge_index, edge_weight, client_features)
                        torch.save(data, osp.join(self.processed_dir, file_name))

                    # for getter function #TODO: fix something better works for now
                    InstanceIdx = [idx2] * len(raw_data["parent_couple_idx"])
                    self.instance_idx.extend(InstanceIdx)


                    # add couple labels
                    self.parent_couple_idx.extend(raw_data['parent_couple_idx'])

                    # save labels:
                    labels = self.flatten_labels(raw_data["labels"])
                    self.labels.extend(labels)

                    self.accuracy.extend(raw_data["random_acc"])
                    self.accuracy_limit.extend(raw_data["random_acc_limit"])

                    for solution in raw_data["parent_routes"][batch]:

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

        #full_graph_data = torch.load(osp.join(self.processed_dir, f'FullGraph_{instance_idx}.pt'))

        label = self.labels[idx]
        # because of varying sizes of labels. The labels are put in dict so they can be stacked by dataloader

        label = MyLabel(label)
        full_graph_data = 0
        return p1_data, p2_data, full_graph_data, label

    def get_accuracy_scores(self) -> Tuple[float, float]:
        limit_acc = sum(self.accuracy_limit)/len(self)
        random_acc = sum(self.accuracy)/len(self)
        return limit_acc, random_acc