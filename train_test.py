import configparser
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

config = configparser.ConfigParser()
config.read("./train.cfg")
train_config = config['reinforce']

lr = float(train_config["learning_rate"])
batch_size = int(train_config["batch_size"])
epochs_num = int(train_config["epochs_num"])
steps_num = int(train_config["steps_num"])

env = DVRP()
mgr = RouteManager(env)
model = DynamicAttentionModel(mgr.get_feature_dim())
agent = REINFORCE(model, lr, batch_size)

total_episode = 1
total_step = 1
for epoch in range(epochs_num):
    for step in range(steps_num):
        step_reward = 0
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
                    step_reward += reward
                    writer.add_scalar('episode reward', reward, total_episode)
                    total_episode += 1
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
        
        step_reward /= batch_size
        writer.add_scalar('step reward', step_reward, total_step)
        total_step += 1

        agent.train()
    
