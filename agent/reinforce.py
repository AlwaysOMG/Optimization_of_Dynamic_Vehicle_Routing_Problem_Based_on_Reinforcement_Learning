import torch
import torch.optim as optim

class REINFORCE:
    route_prob_list = None
    episode_trajectory = []
    batch_loss = 0
    
    total_reward = 0
    baseline = 0

    def __init__(self, model, lr, batch_size):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.batch_size = batch_size        

    def get_action(self, tensor, info, is_greedy):
        route, prob = self.model(tensor, info, is_greedy)
        self.route_prob_list = prob

        return route
    
    def memory_trajectory(self, info):
        for vehicle_id, route_prob in enumerate(self.route_prob_list):
            target_node = info[0][vehicle_id][0]
            for i, prob_tuple in enumerate(route_prob):
                if prob_tuple[1] == target_node:
                    before_route_prob = route_prob[:i+1]
                    self.episode_trajectory.extend(before_route_prob)
                    break
            route_prob.clear()
    
    def set_total_reward(self, total_reward):
        self.total_reward = total_reward

    def set_baseline(self, total_reward):
        self.baseline = total_reward

    def cal_loss(self):
        trajectory_log_prob = torch.sum(torch.log(torch.stack(
            [prob[0][index] for prob, index in self.episode_trajectory])))
        loss = - (self.total_reward - self.baseline) * trajectory_log_prob
        self.batch_loss += loss

        self.episode_trajectory = []

    def update_parameter(self):
        loss = self.batch_loss / self.batch_size
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.batch_loss = 0

    def save_model(self, epoch):
        self.model.save_model(epoch)
