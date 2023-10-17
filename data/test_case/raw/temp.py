from pprint import pprint
import pickle
import sys
from itertools import permutations
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
file_name = 'batch_12_rawdata.pkl'
with open(file_name, "rb") as file:
    results = pickle.load(file)

a = results['labels'][6]

b = (np.where(a>0, a, 0))

pprint(a[b.nonzero()])

