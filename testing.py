import numpy as np
import torch
import matplotlib.pyplot as plt

p1_sum_of_routes = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
p2_sum_of_routes = torch.tensor([[3], [4]])


a, b = torch.broadcast_tensors(p1_sum_of_routes[:, None],
                               p2_sum_of_routes[None, :])
test = torch.cat((a, b), -1)


numR_P1 = 36
numR_P2 = 36
Max_to_move = 36
label_shape = (Max_to_move, numR_P1, numR_P2)
label = []

i = 0
for numRoutesMove in range(1, Max_to_move):
    for idx1 in range(0, numR_P1):
        for idx2 in range(0, numR_P2):
            label.append(i)
            i += 1

label_tensor = torch.tensor(label)

print(label_shape)
print(label_tensor)
d_tensor = label_tensor.view(label_shape)

np_tensor = d_tensor.numpy()

move = np.indices(np_tensor.shape)[0]
p1 = np.indices(np_tensor.shape)[1]
p2 = np.indices(np_tensor.shape)[2]
col = np_tensor.flatten()
print(col)
