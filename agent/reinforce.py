import torch
import torch.optim as optim

from model.new_dynamic_attention_model.dynamic_attention_model import DynamicAttentionModel

class REINFORCE:
    route_prob_list = None
    episode_trajectory = []
    batch_loss = 0
    
    total_reward = 0
    baseline = 0

    def __init__(self, parameter_dir=None, lr=1e-4, batch_size=128):
        self.model = DynamicAttentionModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size

        if parameter_dir:
            self.load_model(parameter_dir)

    def get_action(self, obs, is_greedy):
        tensor, info = self.obs_to_tensor(obs)
        route, prob = self.model(tensor, info, is_greedy)
        self.route_prob_list = prob

        return route
    
    def obs_to_tensor(self, obs):
        vehicle_obs = obs[0]
        node_obs = obs[1]
        road_obs = obs[2]
        current_time = obs[3]
        
        static_tensor = torch.tensor([node[:-1] for node in node_obs])
        dynamic_tensor = torch.tensor(road_obs, dtype=torch.float32)

        vehicle_info = vehicle_obs                          # loc_node_id, capacity
        node_info = [[row[2], row[-1]] for row in node_obs] # demand, is_served

        return [static_tensor, dynamic_tensor], [vehicle_info, node_info, current_time]
    
    def memory_trajectory(self, obs):
        _, info = self.obs_to_tensor(obs)
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
            [tup[0] for tup in self.episode_trajectory])))
        loss = - (self.total_reward - self.baseline) * trajectory_log_prob
        self.batch_loss += loss

        self.episode_trajectory = []

    def update_parameter(self, show_grad=False):
        loss = self.batch_loss / self.batch_size
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        if show_grad:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    print(f'Gradient norm for {name}: {grad_norm}')
        
        self.batch_loss = 0

    def save_model(self, epoch):
        self.model.save_model(epoch)
    
    def load_model(self, file_dir):
        self.model.load_state_dict(torch.load(file_dir))