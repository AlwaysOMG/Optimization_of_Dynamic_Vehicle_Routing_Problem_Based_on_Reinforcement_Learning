import configparser
import copy
import torch

from dvrp.dvrp import DVRP
from agent.reinforce import REINFORCE
from utils.writer import Writer # tensorboard --logdir='log'

# load parameters
config = configparser.ConfigParser()
config.read("./config.cfg")
lr = float(config['train']["learning_rate"])
batch_size = int(config['train']["batch_size"])
epochs_num = int(config['train']["epochs_num"])
steps_num = int(config['train']["steps_num"])

# init instance
env = DVRP()
parameter_dir = './model/new_dynamic_attention_model/parameter/50/128_3_4_5.pth'
agent = REINFORCE(parameter_dir, lr, batch_size)
writer = Writer()

# training
for epoch in range(epochs_num):
    for step in range(steps_num):
        for batch in range(batch_size):
            obs = env.reset()
            copy_env = copy.deepcopy(env)
            while True:
                action = agent.get_action(obs, False)
                obs, reward, is_done = env.step(action)
                agent.memory_trajectory(obs)

                if is_done:
                    agent.set_total_reward(reward)
                    episode_travel_cost = env.get_travel_cost()
                    episode_penalty_cost = env.get_penalty_cost()
                    writer.episode_record(episode_travel_cost, episode_penalty_cost, reward)
                    break
            
            # baseline
            with torch.no_grad():
                obs = copy_env.get_observation()
                while True:
                    action = agent.get_action(obs, True)
                    obs, reward, is_done = copy_env.step(action)
    
                    if is_done:
                        agent.set_baseline(reward)
                        break
            
            agent.cal_loss()
        
        agent.update_parameter()
        writer.step_record()
    
    agent.save_model(epoch)
    writer.epoch_record()
