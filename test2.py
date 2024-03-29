"""
This file is used to test the optimization of the dynamic vehicle routing problem.
"""

from model.dvrp import Env
from opt_method.iterated_local_search import iterated_local_search

file_path = "./instance/Vrp-Set-Solomon/R101.txt"
dvrp = Env(file_path, 0.1, 0)
state = dvrp.reset()
while(True):    
    routes = iterated_local_search(state[1][0], state[1][1], state[1][2])
    action = (True, routes)
    state, reward, is_terminate = dvrp.step(action)

    if is_terminate:
        break
