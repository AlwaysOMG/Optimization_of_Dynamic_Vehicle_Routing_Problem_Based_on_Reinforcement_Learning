import torch
import configparser
import time
from tqdm import trange

from dvrp.dvrp import DVRP
from dvrp.route_manager import RouteManager
from model.new_dynamic_attention_model.dynamic_attention_model import DynamicAttentionModel
from agent.reinforce import REINFORCE
from utils.writer import Writer
from agent.aco import ACO
from agent.alns_agent import ALNS_Solver

# load parameters
config = configparser.ConfigParser()
config.read("./config.cfg")
test_config = config['test']
test_instance = int(test_config["test_instance"])
customer_num = int(config["instance"]["customer_num"])

# init instance
env = DVRP()
mgr = RouteManager(env)
model = DynamicAttentionModel(customer_num, mgr.get_feature_dim())
model.load_state_dict(torch.load('./model/new_dynamic_attention_model/parameter/50/parameter_analysis/128_1e-5.pth'))
agent = REINFORCE(model, 0, 0)
writer = Writer(customer_num, is_test=True)

# testing
for i in trange(test_instance):
    start_time = time.time()
    obs = env.reset()
    while True:
        """
        obs_tensor, obs_info = mgr.obs_to_tensor(obs)
        action = agent.get_action(obs_tensor, obs_info, True)
        route = mgr.action_to_route(action)
        obs, reward, is_done = env.step(route)
        """
        
        t = env.get_current_time()
        sol = ALNS_Solver(obs, t).run()
        route = mgr.action_to_route(sol)
        obs, reward, is_done = env.step(route)
        
        
        if is_done:
            break
    end_time = time.time()
    writer.test_record(-reward, end_time-start_time)
writer.test_csv()