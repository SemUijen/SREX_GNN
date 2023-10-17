from pprint import pprint
import pickle
from itertools import permutations

file_name = 'batch_812_rawdata.pkl'
with open(file_name, "rb") as file:
    results = pickle.load(file)

print(results['parent_couple_idx'])

list1 = [53,54,55,56]
new = list(map(tuple, permutations(list1, r=2)))

results['parent_couple_idx'] = new

print(results['parent_couple_idx'])

with open(file_name, "wb") as file:
    pickle.dump(results, file)