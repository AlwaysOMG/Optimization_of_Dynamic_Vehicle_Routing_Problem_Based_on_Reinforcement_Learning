import copy
from tqdm import tqdm

from dvrp.dvrp import DVRP
from dvrp.route_manager import RouteManager
from model.new_dynamic_attention_model.dynamic_attention_model import DynamicAttentionModel
from agent.ppo import PPO

lr = 1e-4
batch_size = 4
steps_num = 10
epochs_num = 1

import configparser
config = configparser.ConfigParser()
config.read("./config.cfg")
customer_num = int(config["instance"]["customer_num"])

env = DVRP()
mgr = RouteManager(env)
model = DynamicAttentionModel(customer_num, mgr.get_feature_dim())
agent = PPO(model, lr, batch_size)

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
                agent.record_serveded_customer(obs_info)

                if is_done:
                    agent.memory()
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

        agent.update_parameter()
