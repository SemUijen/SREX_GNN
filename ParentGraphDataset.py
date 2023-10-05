import os.path as osp
import pickle
from typing import Union, List, Tuple, Callable

import torch
from torch import Tensor
from torch_geometric.data import Dataset
from GraphData import FullGraph, ParentGraph
from data.utils import SolutionTransformer


# from GraphDataLoader import GraphData


class ParentGraphsDataset(Dataset):
    def __init__(self, root: str, is_processed: bool, pre_transform: SolutionTransformer = SolutionTransformer(),
                 transform=None, pre_filter=None):
        self.processed_files = []
        self.is_processed = is_processed
        self.parent_couple_idx = []
        self.instance_idx = []
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> List[str]:
        return ["X-n439-k37_TestSet_v2.pkl"]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self.processed_files

    def read_pickle(self, raw_path):
        with open(raw_path, "rb") as file:
            raw_data = pickle.load(file)
        return raw_data

    def process(self) -> None:
        if not self.is_processed:
            idx = 0
            for raw_path in self.raw_paths:
                # Read data from `raw_path`.
                raw_data = self.read_pickle(raw_path)

                # process_whole_graph
                route_instance = raw_data['route_instance']
                file_name = f'FullGraph_{idx}.pt'
                if self.pre_transform:
                    graph_input, client_features = self.pre_transform(instance_name=route_instance, get_full_graph=True)
                    data = FullGraph(graph_input, client_features)
                    torch.save(data, osp.join(self.processed_dir, file_name))

                # for getter function #TODO: fix something better works for now
                InstanceIdx = [idx] * len(raw_data["parent_couple_idx"])
                self.instance_idx.extend(InstanceIdx)

                # add couple labels
                self.parent_couple_idx.extend(raw_data['parent_couple_idx'])

                # save labels:
                file_name = f'labels_{idx}.pt'
                torch.save(raw_data["labels"], osp.join(self.processed_dir, file_name))

                idx += 1
                for data in raw_data["parent_routes"]:

                    if self.pre_transform:
                        graph_input = self.pre_transform(instance_name=route_instance, get_full_graph=False,
                                                         parent_route=data)
                        data = ParentGraph(graph_input, client_features)
                        file_name = f'ParentGraphs_{idx}.pt'
                        self.processed_files.append(file_name)
                        torch.save(data, osp.join(self.processed_dir, file_name))
                        idx += 1

                    else:
                        raise "No pre_transform"

    def len(self) -> int:
        return len(self.processed_file_names)

    # TODO: fix get function
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'ParentGraphs_{idx}.pt'))
        return data


dataset = ParentGraphsDataset(root="C:/SREX_GNN/data/test_case", is_processed=False)


