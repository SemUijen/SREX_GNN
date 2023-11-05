from pyvrp import read, Solution, ProblemData
import numpy as np
from typing import Tuple, List
import torch
from torch import Tensor


# from GraphDataLoader import GraphData

class SolutionTransformer:

    @staticmethod
    def get_instance(instance_name: str) -> ProblemData:
        if  instance_name in ["R2_8_9", 'R1_4_10']:
            instance = read(f"./data/routes/{instance_name}.vrp", round_func='round', instance_format="solomon")

        else:
            instance = read(f"./data/routes/{instance_name}.vrp", round_func='round')
            instance.client(1).tw_late

        return instance

    @staticmethod
    def route_to_solutions_object(route, instance: ProblemData) -> Solution:
        return Solution(data=instance, routes=route)

    @staticmethod
    def get_node_features(instance: ProblemData) -> Tensor:
        client_features = []

        capacity = instance.vehicle_type(0).capacity
        num_veh = instance.num_vehicles

        for client_nr in range(1, instance.num_clients + 1):
            client = instance.client(client_nr)
            client_features.append(
                [client.x, client.y, client.demand, capacity, num_veh])


        return torch.tensor(client_features, dtype=torch.float)


    @staticmethod
    def get_edge_features_from_instance(instance: ProblemData) -> Tensor:
        return torch.tensor(instance.distance_matrix())

    @staticmethod
    def get_adj_matrix_from_solution(solution: Solution) -> Tensor:
        num_nodes = len(solution.get_neighbours())
        neighbours1 = solution.get_neighbours()
        graph_edge_matrix_sol1 = np.zeros(shape=(num_nodes-1, num_nodes-1), dtype="int64")

        for i in range(1, num_nodes-1):
            fN_sol1, sN_sol1 = neighbours1[i]
            if fN_sol1-1 >= 0:
                graph_edge_matrix_sol1[(fN_sol1-1, i-1)] = 1
            if sN_sol1-1 >= 0:
                graph_edge_matrix_sol1[(i-1, sN_sol1-1)] = 1


        return torch.tensor(graph_edge_matrix_sol1)

    @staticmethod
    def get_client_to_route_vector(solution: Solution) -> Tensor:
        vector = np.zeros(shape=(solution.num_clients()))
        route_nr = 0

        for route in solution.get_routes():
            for client in route:

                vector[client-1] = route_nr

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

        #edge_weight
        row, col = edge_index
        edge_weight = edge_features[1:, 1:][row, col]
        # total number of routes
        num_routes = solution.num_routes()
        client_features = self.get_node_features(instance)

        return client_route_vector, edge_index, edge_weight, num_routes, client_features

    def full_graph_to_input(self, instance: ProblemData):

        client_features = self.get_node_features(instance)
        edge_features = self.get_edge_features_from_instance(instance)

        # number of nodes are total clients + depot
        num_nodes = instance.num_clients + 1
        fully_connected = np.ones(shape=(num_nodes-1, num_nodes-1), dtype="int64")
        fully_connected = torch.tensor(fully_connected)

        edge_index = fully_connected.nonzero().t()
        row, col = edge_index
        edge_weight = edge_features[1:, 1:][row, col]

        return edge_index, edge_weight, client_features

    def __call__(self, instance_name: str, get_full_graph: bool, parent_solution: Solution = None):

        if get_full_graph:
            instance = self.get_instance(instance_name=instance_name)
            edge_index, edge_weight, client_features = self.full_graph_to_input(instance)
            return edge_index, edge_weight, client_features

        else:
            if parent_solution:
                instance = self.get_instance(instance_name=instance_name)
                #solution = self.route_to_solutions_object(route=parent_route, instance=instance)

                return self.solution_to_input(instance=instance, solution=parent_solution)

            else:
                raise "Solution Transformer Expects a Route"
