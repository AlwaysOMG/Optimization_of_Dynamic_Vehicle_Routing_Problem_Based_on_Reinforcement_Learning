import copy
import torch
from tqdm import tqdm

from dvrp.dvrp import DVRP
from agent.reinforce import REINFORCE

parameter_file = None
lr = 1e-4
batch_size = 4
steps_num = 10
epochs_num = 1

env = DVRP()
agent = REINFORCE(parameter_file, lr, batch_size)

for epoch in range(epochs_num):
    for step in tqdm(range(steps_num)):
        for batch in range(batch_size):
            obs = env.reset()
            copy_env = copy.deepcopy(env)

            while True:
                action = agent.get_action(obs, False)
                obs, reward, is_done = env.step(action)
                agent.memory_trajectory(obs)

                if is_done:
                    agent.set_total_reward(reward)
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
        
        agent.update_parameter(show_grad=True)
