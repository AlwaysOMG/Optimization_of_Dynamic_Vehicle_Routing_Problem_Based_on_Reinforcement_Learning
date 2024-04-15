import copy
import datetime
from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir='log'

from dvrp.dvrp import DVRP
from dvrp.route_manager import RouteManager
from model.new_dynamic_attention_model.dynamic_attention_model import DynamicAttentionModel
from agent.reinforce import REINFORCE

date = datetime.date.today()
time = datetime.datetime.now().strftime("%H-%M-%S")
writer = SummaryWriter(log_dir=f"log/{date}_{time}")

env = DVRP()
mgr = RouteManager(env)
model = DynamicAttentionModel(mgr.get_feature_dim())
agent = REINFORCE(model, 0.00001)

for epoch in range(30):
    for step in range(10000):
        step_reward = 0
        for batch in range(128):
            obs = env.reset()
            copy_env = copy.deepcopy(env)
            copy_mgr = RouteManager(copy_env)
            while True:
                obs_tensor, obs_info = mgr.obs_to_tensor(obs)
                action = agent.get_action(obs_tensor, obs_info, False)
                route = mgr.action_to_route(action)
                obs, reward, is_done = env.step(route)
    
                _, obs_info = mgr.obs_to_tensor(obs)
                agent.memory_trajectory(action, obs_info)

                if is_done:
                    agent.set_total_reward(reward)
                    agent.reset_trajectory()
                    step_reward += reward
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
        
        step_reward /= 128
        writer.add_scalar('step reward', step_reward, step)

        agent.train()
        
        
            
            

            

