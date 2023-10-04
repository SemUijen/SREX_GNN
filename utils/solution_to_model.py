import os
from pyvrp import read, Solution, ProblemData
import vrplib
import numpy as np
from typing import Tuple
import torch


def get_route_instance():
    instance_name = 'X-n439-k37'
    instance = read(f"./data/{instance_name}/{instance_name}.vrp", round_func='round')
    instance_bks = vrplib.read_solution(f"./data/{instance_name}/{instance_name}.sol")
    instance.client(1).tw_late

    # TODO: Get Customer Node Features node features

    return instance


def get_node_features_from_instance(route_instance: ProblemData):
    client_features = []

    for client_nr in range(route_instance.num_clients + 1):
        client = route_instance.client(client_nr)
        client_features.append(
            [client.x, client.y, client.tw_late, client.tw_early, client.demand, client.service_duration])

    return torch.tensor(client_features, dtype=torch.float)


def get_edge_features_from_instance(route_instance: ProblemData):
    return torch.tensor(route_instance.distance_matrix())


def get_pyvrp_solutions(route1, route2):
    instance = get_route_instance()
    solution1 = Solution(data=instance, routes=route1)
    solution2 = Solution(data=instance, routes=route2)

    return solution1, solution2


def get_example_solutions():
    test_solution_p1 = [[406, 42, 239, 218, 281, 435, 162, 346, 228, 308, 202, 149],
                        [58, 361, 331, 220, 240, 408, 152, 255, 302, 340, 108, 430],
                        [28, 248, 19, 330, 343, 389, 327, 24, 175, 401],
                        [132, 4, 34, 230, 21, 181, 84, 382, 192, 235, 165, 81],
                        [73, 157, 135, 177, 67, 16, 112, 378, 11, 244, 292, 174],
                        [155, 41, 275, 92, 348, 169, 2, 8, 311, 434, 105, 26],
                        [44, 121, 325, 249, 433, 337, 391, 423, 396, 420, 413, 243],
                        [172, 71, 335, 72, 97, 267, 349, 223, 425, 133, 370, 3],
                        [411, 400, 139, 200, 145, 206, 409, 110, 245, 241, 393, 237],
                        [250, 211, 347, 43, 375, 296, 57, 392, 299, 386, 410, 260],
                        [268, 377, 342, 360, 242, 372, 264, 352, 86, 315, 432, 437],
                        [431, 173, 88, 210, 22, 56, 75, 196, 140, 62, 287, 207],
                        [303, 225, 388, 83, 416, 407, 366, 122, 418, 421, 381, 270],
                        [404, 166, 345, 385, 438, 312, 280, 221, 227, 380, 324, 233],
                        [283, 144, 334, 25, 124, 99, 77, 399, 106, 216, 96, 179],
                        [146, 189, 61, 130, 80, 344, 197, 7, 118, 91, 154, 65],
                        [351, 266, 329, 319, 309, 101, 252, 338, 289, 253, 257, 323],
                        [402, 428, 403, 384, 17, 412, 89, 293, 339, 66, 126, 297],
                        [137, 159, 15, 153, 215, 193, 285, 271, 47, 376, 138, 246],
                        [422, 353, 195, 115, 229, 321, 383, 350, 286, 341, 98, 251],
                        [180, 107, 306, 53, 20, 111, 148, 314, 85, 12, 48],
                        [204, 79, 131, 6, 1, 397, 387, 134, 426, 5, 31, 333],
                        [176, 367, 274, 116, 113, 90, 109, 371, 390, 184, 395, 161],
                        [209, 354, 74, 63, 117, 103, 13, 394, 171, 356, 290, 368],
                        [364, 259, 30, 190, 125, 222, 310, 178, 291, 160, 313, 265],
                        [199, 39, 232, 35, 322, 374, 208, 141, 27, 100, 51, 164],
                        [379, 301, 37, 10, 64, 123, 272, 357, 363, 424, 114, 316],
                        [282, 224, 273, 188, 369, 336, 304, 419, 405, 234, 320, 78],
                        [256, 182, 269, 284, 150, 205, 219, 142, 151, 163, 183, 170],
                        [417, 300, 214, 198, 52, 168, 95, 328, 213, 279, 120, 332],
                        [128, 186, 54, 45, 94, 212, 46, 49, 167, 277, 69, 68],
                        [156, 36, 398, 373, 436, 14, 261, 262, 307, 18, 50, 194],
                        [355, 298, 23, 70, 191, 129, 258, 93, 226, 317, 415, 427],
                        [254, 9, 429, 263, 55, 288, 365, 359, 295],
                        [38, 203, 201, 76, 119, 29, 158, 87, 102, 127, 40, 247],
                        [276, 278, 33, 318, 294, 185, 104, 60, 143, 147, 82, 32],
                        [414, 231, 358, 238, 305, 362, 187, 136, 59, 236, 217, 326]]
    test_solution_p2 = [[217, 236, 105, 2, 434, 8, 311, 133, 370, 169, 3, 348],
                        [411, 410, 349, 425, 223, 200, 139, 57, 218, 239, 42, 335],
                        [414, 26, 260, 92, 275, 41, 406, 270, 308, 202, 149, 172],
                        [72, 97, 400, 267, 386, 299, 392, 296, 375, 43, 347, 211],
                        [195, 393, 280, 312, 438, 385, 345, 166, 435, 162, 346, 228],
                        [326, 155, 71, 404, 381, 237, 121, 325, 249, 353, 422, 44],
                        [281, 206, 145, 122, 366, 384, 403, 17, 412, 407, 416, 418],
                        [303, 245, 250, 421, 409, 110, 388, 225, 86, 315, 352, 264],
                        [233, 377, 433, 337, 242, 241, 360, 342, 221, 227, 380, 115],
                        [437, 323, 243, 321, 413, 420, 396, 423, 391, 268, 229, 324],
                        [372, 297, 126, 66, 339, 83, 293, 89, 428, 402, 101, 252],
                        [383, 257, 253, 289, 271, 338, 309, 319, 329, 266, 351, 432],
                        [146, 251, 137, 98, 341, 350, 286, 47, 376, 138, 246, 283],
                        [285, 193, 215, 153, 91, 118, 173, 88, 80, 130, 79, 25],
                        [334, 144, 431, 189, 61, 344, 197, 7, 154, 15, 159, 65],
                        [176, 131, 6, 140, 274, 113, 116, 196, 75, 56, 22, 210],
                        [361, 220, 408, 371, 109, 90, 62, 1, 367, 426, 5, 204],
                        [333, 31, 134, 387, 397, 390, 184, 395, 287, 161, 58, 207],
                        [401, 24, 343, 340, 240, 108, 331, 330, 19, 248, 430, 28],
                        [175, 73, 135, 157, 389, 327, 302, 255, 152, 177, 34, 4],
                        [244, 11, 209, 378, 112, 16, 67, 230, 21, 181, 84, 132],
                        [313, 174, 160, 291, 178, 222, 310, 292, 382, 192, 235, 165],
                        [190, 125, 394, 301, 379, 103, 117, 63, 74, 354, 13, 171],
                        [99, 77, 216, 106, 399, 316, 114, 30, 259, 364, 265, 124],
                        [320, 234, 405, 168, 272, 123, 64, 10, 37, 356, 290, 368, 417],
                        [332, 120, 279, 213, 328, 95, 52, 198, 214, 357, 363, 300, 424], [179, 96, 256, 182, 78],
                        [150, 284, 269, 273, 188, 224, 282, 219, 205, 183, 170, 81],
                        [142, 369, 336, 419, 304, 322, 374, 208, 141, 51, 164, 186],
                        [53, 163, 151, 199, 128, 277, 69, 68, 111, 20, 306],
                        [39, 232, 35, 100, 27, 54, 45, 94, 212, 46, 49, 167],
                        [180, 48, 12, 107, 314, 148, 85, 38, 194, 427, 156],
                        [203, 247, 40, 201, 76, 119, 127, 102, 87, 29, 158, 191, 129],
                        [50, 436, 373, 14, 307, 18, 262, 261, 258, 93, 23, 318],
                        [36, 398, 317, 226, 278, 33, 298, 288, 55, 263, 355, 415],
                        [276, 294, 185, 70, 104, 60, 143, 147, 82, 32, 305, 238],
                        [231, 59, 136, 187, 362, 358, 295, 359, 365, 429, 9, 254]]

    pyvrp_solution1, pyvrp_solution2 = get_pyvrp_solutions(test_solution_p1, test_solution_p2)

    return pyvrp_solution1, pyvrp_solution2


def get_adj_matrix_from_solutions(solutions: Tuple[Solution, Solution]):
    solution1, solution2 = solutions
    num_nodes = len(solution1.get_neighbours())
    neighbours1 = solution1.get_neighbours()
    neighbours2 = solution2.get_neighbours()
    graph_edge_matrix_sol1 = np.zeros(shape=(num_nodes, num_nodes), dtype="int64")
    graph_edge_matrix_sol2 = np.zeros(shape=(num_nodes, num_nodes), dtype="int64")

    for i in range(1, num_nodes):
        fN_sol1, sN_sol1 = neighbours1[i]
        fN_sol2, sN_sol2 = neighbours2[i]

        # solution 1
        graph_edge_matrix_sol1[(fN_sol1, i)] = 1
        graph_edge_matrix_sol1[(i, sN_sol1)] = 1
        # solution 2
        graph_edge_matrix_sol2[(fN_sol2, i)] = 1
        graph_edge_matrix_sol2[(i, sN_sol2)] = 1

    return torch.tensor(graph_edge_matrix_sol1), torch.tensor(graph_edge_matrix_sol2)


def get_client_to_route_vector(solution: Solution):
    vector = np.zeros(shape=(solution.num_clients() + 1))
    route_nr = 0

    for route in solution.get_routes():
        for client in route:
            vector[client] = route_nr

        route_nr += 1

    return torch.tensor(vector, dtype=torch.int)


def solutions_to_model(route_instance: ProblemData, parents: Tuple[Solution, Solution]):
    sol1, sol2 = parents

    # client_to_route_vectors:
    sol1_client_route_vector = get_client_to_route_vector(sol1)
    sol2_client_route_vector = get_client_to_route_vector(sol2)

    # edge_index: adjacency matrix of solution -> edge_index(COO format)
    adj_sol1, adj_sol2 = get_adj_matrix_from_solutions(parents)
    sol1_edge_index = adj_sol1.nonzero().t()
    sol2_edge_index = adj_sol2.nonzero().t()

    # x
    client_features = get_node_features_from_instance(route_instance)

    # edge_attr
    edge_features = get_edge_features_from_instance(route_instance)
    ## parent1
    row, col = sol1_edge_index
    sol1_edge_weight = edge_features[row, col]
    ## parent2
    row2, col2 = sol2_edge_index
    sol2_edge_weight = edge_features[row2, col2]

    # total number of routes
    sol1_num_routes = sol1.num_routes()
    sol2_num_routes = sol2.num_routes()

    sol1_input = (sol1_client_route_vector, sol1_edge_index, sol1_edge_weight, sol1_num_routes)
    sol2_input = (sol2_client_route_vector, sol2_edge_index, sol2_edge_weight, sol2_num_routes)

    return sol1_input, sol2_input, client_features
