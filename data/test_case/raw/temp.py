from pprint import pprint
import pickle
import sys

import torch
from itertools import permutations
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from plotting.plot_srex_parameters import plot_srex_parameters
file_name = 'batch_cvrp_6_rawdata.pkl'
with open(file_name, "rb") as file:
    results = pickle.load(file)

print(results.keys())
a = results['labels'][0][2]

print(len(a))
print(results['random_acc_limit'])
print(results['parent_routes'][0][3].num_routes())
print(results['parent_routes'][0][0].num_routes())
label_tensor = torch.tensor(a)
d_tensor = label_tensor.view(24,24,37)
plot_srex_parameters(d_tensor.numpy())