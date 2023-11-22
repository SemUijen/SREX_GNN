import os.path as osp
import pickle
from typing import Union, List, Tuple

import torch
from torch_geometric.data import Dataset
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
from data.utils.GraphData import FullGraph, ParentGraph
from data.utils import SolutionTransformer
from data.utils.Normalize import normalize_graphs
k = 6
PE = AddLaplacianEigenvectorPE(k, attr_name=None, is_undirected=True)

class MyLabel:
    def __init__(self, label):
        self.label = label


class ParentGraphsDataset(Dataset):
    def __init__(self, root: str, raw_files: List[str], instances: List[str], use_instances: List[str], is_processed: bool = False,
                 use_time: bool = False, pre_transform: SolutionTransformer = SolutionTransformer(),
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
        self.use_instances = use_instances
        self.use_time = use_time
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

        # First get FullGraphs
        idx = 0
        for instance in self.instances:
            edge_index, edge_weight, client_features, client_demand, client_pos, depot_pos, client_time = self.pre_transform(instance_name=instance,
                                                                                                                get_full_graph=True)

            data = FullGraph(edge_index, edge_weight, client_features, client_demand, client_pos, depot_pos, client_time)
            data = normalize_graphs(data, TW=self.use_time)
            data = PE(data)
            max_distance = edge_weight.max().item()
            file_name = f'FullGraph_{idx}.pt'
            torch.save(data, osp.join(self.processed_dir, file_name))
            self.instance_dict[instance] = idx, max_distance

            idx += 1

        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            raw_data = self.read_pickle(raw_path)

            for batch in range(len(raw_data["parent_routes"])):
                # process_whole_graph
                route_instance = raw_data["instances"][batch]


                if route_instance in self.use_instances:
                    # for getting the correct Full_graph
                    InstanceIdx = [self.instance_dict[route_instance][0]] * 12
                    self.instance_idx.extend(InstanceIdx)

                    # add couple labels
                    self.parent_couple_idx.extend(raw_data['parent_couple_idx'][batch])
                    self.labels.extend(raw_data["labels"][batch])
                    self.accuracy.extend(raw_data["random_acc"][batch])
                    self.accuracy_limit.extend(raw_data["random_acc_limit"][batch])

                    if not self.is_processed:
                        for i in range(len(raw_data["parent_routes"][batch])):
                            solution = raw_data["parent_routes"][batch][i]
                            idx = raw_data["parent_ids"][batch][i]
                            max_dis = self.instance_dict[route_instance][1]
                            InstanceIdx = self.instance_dict[route_instance][0]
                            data = ParentGraph(*self.pre_transform(instance_name=route_instance,
                                                                  get_full_graph=False, parent_solution=solution))

                            data = normalize_graphs(data, max_distance=max_dis, TW=self.use_time)

                            graph = torch.load(osp.join(self.processed_dir, f'FullGraph_{InstanceIdx}.pt'))
                            data.x = torch.cat([data.x, graph.x[:,-k:]], dim=-1)
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

        #full_graph = torch.load(osp.join(self.processed_dir, f'FullGraph_{instance_idx}.pt'))

        label = self.labels[idx]
        # because of varying sizes of labels. The labels are put in dict so they can be stacked by dataloader

        label = MyLabel(label)

        acc = self.accuracy[idx]

        return p1_data, p2_data, label, instance_idx, torch.tensor(acc)

    def get_accuracy_scores(self) -> str:
        limit_acc = sum(self.accuracy_limit) / len(self)
        random_acc = sum(self.accuracy) / len(self)

        pos_acc = len(torch.nonzero(torch.tensor(self.accuracy))) / len(self)
        lim_pos_acc = len(torch.nonzero(torch.tensor(self.accuracy_limit))) / len(self)

        return ("\n"
                F" random acc (pos):        {random_acc} \n"
                F" lim random acc (pos):    {limit_acc} \n"
                F" Parents with improv:     {pos_acc} \n"
                F" lim Parent with improv:  {lim_pos_acc}")

    