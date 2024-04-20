import configparser
import copy
from tqdm import tqdm
import time

from dvrp.dvrp import DVRP
from dvrp.route_manager import RouteManager
from model.new_dynamic_attention_model.dynamic_attention_model import DynamicAttentionModel
from agent.reinforce import REINFORCE
from utils.writer import Writer

config = configparser.ConfigParser()
config.read("./config.cfg")
train_config = config['train']

lr = float(train_config["learning_rate"])
batch_size = int(train_config["batch_size"])
epochs_num = int(train_config["epochs_num"])
steps_num = int(train_config["steps_num"])
test_instance = int(train_config["test_instance"])

env = DVRP()
mgr = RouteManager(env)
model = DynamicAttentionModel(mgr.get_feature_dim())
agent = REINFORCE(model, lr, batch_size)
writer = Writer()

# training
for epoch in range(epochs_num):
    for step in range(steps_num):
        for batch in tqdm(range(batch_size)):
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
                    writer.episode_record(env.get_travel_cost(), env.get_penalty_cost(), reward)
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
        writer.step_record()
    
    agent.save_model(epoch)
    writer.epoch_record()

# testing
for i in range(test_instance):
    start_time = time.time()
    obs = env.reset()
    while True:
        obs_tensor, obs_info = mgr.obs_to_tensor(obs)
        action = agent.get_action(obs_tensor, obs_info, False)
        route = mgr.action_to_route(action)
        obs, reward, is_done = env.step(route)
    
        if is_done:
            writer.episode_record(env.get_travel_cost(), env.get_penalty_cost(), reward)
            break
    end_time = time.time()
    writer.test_record(-reward, end_time-start_time)
writer.test_csv()