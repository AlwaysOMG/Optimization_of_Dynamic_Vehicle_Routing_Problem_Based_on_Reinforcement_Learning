import torch
import configparser
import time
from tqdm import trange

from dvrp.dvrp import DVRP
from dvrp.route_manager import RouteManager
from model.new_dynamic_attention_model.dynamic_attention_model import DynamicAttentionModel
from agent.reinforce import REINFORCE
from utils.writer import Writer
from agent.alns_agent import ALNS_Solver
from agent.ga import GA
from agent.bso_aco import BSO_ACO

# load parameters
config = configparser.ConfigParser()
config.read("./config.cfg")
test_config = config['test']
test_instance = int(test_config["test_instance"])
customer_num = int(config["instance"]["customer_num"])

# init instance
env = DVRP()
mgr = RouteManager(env)
model = DynamicAttentionModel(mgr.get_feature_dim())
model.load_state_dict(torch.load('./model/new_dynamic_attention_model/parameter/50/origin.pth'))
agent = REINFORCE(model, 0, 0)
writer = Writer(is_test=True)

# testing
for i in trange(test_instance):
    start_time = time.time()
    obs = env.reset()
    while True:
        obs_tensor, obs_info = mgr.obs_to_tensor(obs)
        action = agent.get_action(obs_tensor, obs_info, True)
        route = mgr.action_to_route(action)
        obs, reward, is_done = env.step(route)


        """
        sol = ALNS_Solver(obs).run()
        route = mgr.action_to_route(sol)
        obs, reward, is_done = env.step(route)
        
        """
        
        if is_done:
            break
    end_time = time.time()
    writer.test_record(-reward, end_time-start_time)
writer.test_csv()