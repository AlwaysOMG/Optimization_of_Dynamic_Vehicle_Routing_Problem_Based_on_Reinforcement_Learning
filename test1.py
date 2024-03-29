"""
This file is used to test the performance of iterated local search.
"""

from model.dvrp import Env
from opt_method.utils import cal_cost
from opt_method.iterated_local_search import iterated_local_search

file_path = "./instance/Vrp-Set-Solomon/R101.txt"
dvrp = Env(file_path, 0, 0)
state = dvrp.reset()
routes = iterated_local_search(state[1][0], state[1][1], state[1][2])
cost = sum(cal_cost(r, state[0], state[1], state[2]) for r in routes)
print(routes)
print(cost)
