import time

from dvrp.dvrp import DVRP
from dvrp.route_manager import RouteManager
from model.new_dynamic_attention_model.dynamic_attention_model import DynamicAttentionModel
from agent.alns_agent import ALNS_Solver

import configparser
config = configparser.ConfigParser()
config.read("./config.cfg")
customer_num = int(config["instance"]["customer_num"])

env = DVRP()
mgr = RouteManager(env)
model = DynamicAttentionModel(customer_num, mgr.get_feature_dim())

obs = env.reset()
while True:
    sol = ALNS_Solver(obs).run()
    route = mgr.action_to_route(sol)
    obs, reward, is_done = env.step(route)

    if is_done:
        print(reward)
        break
