from pprint import pprint
import pickle

with open('X-n439-k37_TestSet.pkl', "rb") as file:
    results = pickle.load(file)

instance = results[0]["route_instance"]
parent_routes = [results[0]['parent1_route'], results[0]['parent1_route'], results[1]['parent1_route']]
parent_couple_idx = [(1, 2), (1, 3), (2, 3)]


labels = [results[0]['label'], results[1]['label'], results[1]['label']]

data = {
    "route_instance": instance,
    "parent_routes": parent_routes,
    "parent_couple_idx": parent_couple_idx,
    "labels": labels,
}

with open('X-n439-k37_TestSet_v2.pkl', "wb") as file:
    pickle.dump(data, file)
