import numpy as np
import torch
import numpy

"""p1_sum_of_routes = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
p2_sum_of_routes = torch.tensor([[3], [4]])

print(p1_sum_of_routes.shape)
a, b = torch.broadcast_tensors(p1_sum_of_routes[:, None],
                               p2_sum_of_routes[None, :])
test = torch.cat((a, b), -1)

print(test)
print(test.flatten(0, 1))"""

numR_P1 = 3
numR_P2 = 3
Max_to_move = 3
label_shape = (Max_to_move - 1, numR_P1, numR_P2)
label = []

i = 0
for numRoutesMove in range(1, Max_to_move):
    for idx1 in range(0, numR_P1):
        for idx2 in range(0, numR_P2):
            label.append(i)
            i += 1

label_tensor = torch.tensor(label)


def best_to_srexParameters(best_idx, label_shape):
    print(best_idx, label_shape)
    Max_to_move, numR_P1, numR_P2 = label_shape
    i = 0
    for numRoutesMove in range(1, Max_to_move+1):
        for idx1 in range(0, numR_P1):
            for idx2 in range(0, numR_P2):
                print(i)
                if i == best_idx:
                    return numRoutesMove, idx1, idx2
                i += 1

    return "nothing found"
