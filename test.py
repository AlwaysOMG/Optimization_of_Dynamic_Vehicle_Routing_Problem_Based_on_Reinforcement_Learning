import random
import numpy as np
import configparser
import time
from tqdm import trange

from dvrp.dvrp import DVRP
from agent.reinforce import REINFORCE
from meta_heuristics.alns_agent import ALNS_Solver
from meta_heuristics.ga import GA
from meta_heuristics.bso_aco import BSO_ACO
from utils.writer import Writer
from utils.plotter import Plotter

# random seed
seed = 1024
random.seed(seed)
np.random.seed(seed)

# load parameters
config = configparser.ConfigParser()
config.read("./config.cfg")
test_instance = int(config['test']["test_instance"])
test_instance_seed = random.sample(range(0, 1024), 100)

# initialize
parameter_dir = './model/peng_dynamic_attention_model/parameter/final.pth'
env = DVRP()
agent = REINFORCE(parameter_dir)
writer = Writer(is_test=True)
plotter = Plotter()

# testing
is_plot = False
is_instance_analysis = False
for i in trange(test_instance):
    start_time = time.time()
    obs = env.reset(test_instance_seed[i])
    while True:
        action = agent.get_action(obs, True)
        #action = ALNS_Solver(obs).run()
        obs, reward, is_done = env.step(action)
        
        if is_done:
            break
    end_time = time.time()
    
    episode_travel_cost = env.get_travel_cost()
    episode_penalty_cost = env.get_penalty_cost()
    episode_service_status = env.get_service_status()
    writer.test_record(-reward, episode_travel_cost, episode_penalty_cost, 
                       episode_service_status, end_time-start_time)
    if is_plot:
        node_data = obs[1]
        vehicle_route = env.get_service_list()
        plotter.plot(i, node_data, episode_service_status, vehicle_route)
    
    if is_instance_analysis:
        node_data = obs[1]
        writer.instance_analysis(node_data)
