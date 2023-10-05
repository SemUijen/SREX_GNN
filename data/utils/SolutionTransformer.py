import vrplib
from pyvrp import read, Solution, ProblemData
import numpy as np
from typing import Tuple, List
import torch
from torch import Tensor


# from GraphDataLoader import GraphData

class SolutionTransformer:

    @staticmethod
    def get_instance(instance_name: str) -> ProblemData:
        instance = read(f"./data/routes/{instance_name}.vrp", round_func='round')

        # TODO: Add instance type to ProblemData
        # if instance.type is not 'VRPTW':
        #     instance.client(1).tw_late
        instance.client(1).tw_late

        return instance

    @staticmethod
    def route_to_solutions_object(route, instance: ProblemData) -> Solution:
        return Solution(data=instance, routes=route)

    @staticmethod
    def get_node_features(instance: ProblemData) -> Tensor:
        client_features = []

        for client_nr in range(instance.num_clients + 1):
            client = instance.client(client_nr)
            client_features.append(
                [client.x, client.y, client.tw_late, client.tw_early, client.demand, client.service_duration])

        return torch.tensor(client_features, dtype=torch.float)

    @staticmethod
    def get_node_features_from_instance(instance: ProblemData) -> Tensor:
        client_features = []

        for client_nr in range(instance.num_clients + 1):
            client = instance.client(client_nr)
            client_features.append(
                [client.x, client.y, client.tw_late, client.tw_early, client.demand, client.service_duration])

        return torch.tensor(client_features, dtype=torch.float)

    @staticmethod
    def get_edge_features_from_instance(instance: ProblemData) -> Tensor:
        return torch.tensor(instance.distance_matrix())

    @staticmethod
    def get_adj_matrix_from_solution(solution: Solution) -> Tensor:
        num_nodes = len(solution.get_neighbours())
        neighbours1 = solution.get_neighbours()
        graph_edge_matrix_sol1 = np.zeros(shape=(num_nodes, num_nodes), dtype="int64")

        for i in range(1, num_nodes):
            fN_sol1, sN_sol1 = neighbours1[i]

            graph_edge_matrix_sol1[(fN_sol1, i)] = 1
            graph_edge_matrix_sol1[(i, sN_sol1)] = 1

        return torch.tensor(graph_edge_matrix_sol1)

    @staticmethod
    def get_client_to_route_vector(solution: Solution) -> Tensor:
        vector = np.zeros(shape=(solution.num_clients() + 1))
        route_nr = 0

        for route in solution.get_routes():
            for client in route:
                vector[client] = route_nr

            route_nr += 1

        return torch.tensor(vector, dtype=torch.int)

    def solution_to_input(self, instance: ProblemData, solution: Solution):

        # client_to_route_vectors:
        client_route_vector = self.get_client_to_route_vector(solution)

        # edge_index: adjacency matrix of solution -> edge_index(COO format)
        adj_sol = self.get_adj_matrix_from_solution(solution)
        edge_index = adj_sol.nonzero().t()

        # edge_attr
        edge_features = self.get_edge_features_from_instance(instance)
        ## parent1
        row, col = edge_index
        edge_weight = edge_features[row, col]

        # total number of routes
        num_routes = solution.num_routes()

        sol1_input = (client_route_vector, edge_index, edge_weight, num_routes)

        return sol1_input

    def full_graph_to_input(self, instance: ProblemData):

        client_features = self.get_node_features(instance)
        edge_weight = self.get_edge_features_from_instance(instance)

        # number of nodes are total clients + depot
        num_nodes = instance.num_clients + 1
        fully_connected = np.ones(shape=(num_nodes, num_nodes), dtype="int64")
        fully_connected = torch.tensor(fully_connected)

        edge_index = fully_connected.nonzero().t()

        full_graph_input = (edge_index, edge_weight)
        return full_graph_input, client_features

    def __call__(self, instance_name: str, get_full_graph: bool, parent_route: List[List[int]] = None):

        if get_full_graph:
            instance = self.get_instance(instance_name=instance_name)
            graph_input, client_features = self.full_graph_to_input(instance)
            return graph_input, client_features

        else:
            if parent_route:
                instance = self.get_instance(instance_name=instance_name)
                solution = self.route_to_solutions_object(route=parent_route, instance=instance)

                GraphData_input = self.solution_to_input(instance=instance, solution=solution)
                return GraphData_input

            else:
                raise "Solution Transformer Expects a Route"
