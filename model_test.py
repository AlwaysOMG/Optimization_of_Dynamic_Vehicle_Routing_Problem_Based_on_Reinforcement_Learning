import time

from dvrp.dvrp import DVRP
from dvrp.route_manager import RouteManager
from model.new_dynamic_attention_model.dynamic_attention_model import DynamicAttentionModel
from agent.reinforce import REINFORCE
from agent.alns_agent import ALNS_Solver
from agent.ga import GA
from agent.bso_aco import  BSO_ACO

import configparser
config = configparser.ConfigParser()
config.read("./config.cfg")
customer_num = int(config["instance"]["customer_num"])

env = DVRP()
mgr = RouteManager(env)
model = DynamicAttentionModel(customer_num, mgr.get_feature_dim())
agent = REINFORCE(model, 0.001, 128)

obs = env.reset()
while True:
    obs_tensor, obs_info = mgr.obs_to_tensor(obs)
    action = agent.get_action(obs_tensor, obs_info, False)
    route = mgr.action_to_route(action)
    obs, reward, is_done = env.step(route)

    _, obs_info = mgr.obs_to_tensor(obs)
    agent.memory_trajectory(obs_info)

    if is_done:
        agent.cal_loss()
        break
