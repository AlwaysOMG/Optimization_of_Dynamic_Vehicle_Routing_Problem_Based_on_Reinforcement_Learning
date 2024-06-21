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

# random seed
seed = 1024
random.seed(seed)
np.random.seed(seed)

# load parameters
config = configparser.ConfigParser()
config.read("./config.cfg")
test_instance = int(config['test']["test_instance"])

# initialize
env = DVRP()
parameter_dir = './model/new_dynamic_attention_model/parameter/50/dynamic_3.pth'
agent = REINFORCE(parameter_dir)
writer = Writer(is_test=True)

# testing
for i in trange(test_instance):
    start_time = time.time()
    obs = env.reset()
    while True:
        #action = agent.get_action(obs, True)
        action = GA(obs).run()
        obs, reward, is_done = env.step(action)
        
        if is_done:
            break
    end_time = time.time()
    writer.test_record(-reward, end_time-start_time)
writer.test_csv()