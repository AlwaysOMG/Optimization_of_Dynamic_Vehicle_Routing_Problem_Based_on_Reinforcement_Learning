import configparser

import numpy as np
import torch
import torch.optim as optim

config = configparser.ConfigParser()
config.read("./agent/train.cfg")
parameter_config = config['reinforce']

class REINFORCE:
    lr = float(parameter_config["learning_rate"])
    batch_size = int(parameter_config["batch_size"])

    def __init__(self, model, lr):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        self.prob_dict_list = None
        self.trajectory_prob = 1
        self.total_reward = 0
        self.baseline = 0

        self.batch_loss = []

    def get_action(self, tensor, info, is_greedy):
        route, prob = self.model(tensor, info, is_greedy)
        self.prob_dict_list = prob

        return route
    
    def memory_trajectory(self, action, info):
        for vehicle_id, route in enumerate(action):
            target_node = info[0][vehicle_id][0]
            if target_node in route:
                drive_route = route[:route.index(target_node)+1]
                for node_id in drive_route:
                    prob = self.prob_dict_list[vehicle_id][node_id]
                    if prob != 1:
                        self.trajectory_prob *= prob
    
    def set_total_reward(self, total_reward):
        self.total_reward = total_reward

    def set_baseline(self, total_reward):
        self.baseline = total_reward

    def cal_loss(self):
        log_prob = np.log(self.trajectory_prob)
        loss = - (self.total_reward - self.baseline) * log_prob
        self.batch_loss.append(loss)

    def reset_trajectory(self):
        self.trajectory_prob = 1

    def train(self):
        loss = torch.tensor(self.batch_loss, requires_grad=True).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.batch_loss = []
