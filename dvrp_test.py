import numpy as np

from dvrp.dvrp import DVRP
from dvrp.route_manager import RouteManager

env = DVRP()
mgr = RouteManager(env)
env.reset()

route_list = [np.array([1,2,3,4,5,0,6,7,8,9,0]),
              np.array([10,12,13,14,0,15,16,17,18,19,20,11,0])]
r = mgr.action_to_route(route_list)

while not env.check_done():
    obs, reward, done = env.step(r)

print([node.check_served() if node != env.get_depot() else 0 
       for node in env.node_list])