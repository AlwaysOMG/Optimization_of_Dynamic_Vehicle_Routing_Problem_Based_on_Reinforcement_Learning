import copy
from tqdm import tqdm

from dvrp.dvrp import DVRP
from dvrp.route_manager import RouteManager
from model.new_dynamic_attention_model.dynamic_attention_model import DynamicAttentionModel
from agent.reinforce import REINFORCE

lr = 1e-4
batch_size = 128
steps_num = 10
epochs_num = 1

import configparser
config = configparser.ConfigParser()
customer_num = int(config["instance"]["customer_num"])

env = DVRP()
mgr = RouteManager(env)
model = DynamicAttentionModel(customer_num, mgr.get_feature_dim())
agent = REINFORCE(model, lr, batch_size)

for epoch in range(epochs_num):
    for step in tqdm(range(steps_num)):
        for batch in range(batch_size):
            obs = env.reset()
            copy_env = copy.deepcopy(env)
            copy_mgr = RouteManager(copy_env)
            while True:
                obs_tensor, obs_info = mgr.obs_to_tensor(obs)
                action = agent.get_action(obs_tensor, obs_info, False)
                route = mgr.action_to_route(action)
                obs, reward, is_done = env.step(route)
    
                _, obs_info = mgr.obs_to_tensor(obs)
                agent.memory_trajectory(obs_info)

                if is_done:
                    agent.set_total_reward(reward)
                    break

            # baseline
            obs = copy_env.get_observation()
            while True:
                obs_tensor, obs_info = copy_mgr.obs_to_tensor(obs)
                action = agent.get_action(obs_tensor, obs_info, True)
                route = copy_mgr.action_to_route(action)
                obs, reward, is_done = copy_env.step(route)
    
                if is_done:
                    agent.set_baseline(reward)
                    break
            
            agent.cal_loss()

        agent.update_parameter()
