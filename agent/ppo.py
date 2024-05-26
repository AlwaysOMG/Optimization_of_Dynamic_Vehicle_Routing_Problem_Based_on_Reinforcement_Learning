import time
import torch
import torch.optim as optim

class PPO:
    device = torch.device("cuda:0")
    state_trajectory = []
    action_trajectory = []
    served_customer_trajectory = []
    total_reward = 0
    baseline = 0

    state_memory_buffer = []
    action_memory_buffer = []
    served_customer_memory_buffer = []
    advantage_memory_buffer = []

    def __init__(self, model, lr, batch_size):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.reuse_times = 8
        self.eps = 0.2
    
    def get_action(self, tensor, info, is_greedy):
        route, prob = self.model(tensor, info, is_greedy)
        if not is_greedy:
            self.state_trajectory.append((tensor, info))
            self.action_trajectory.append(prob)

        return route
    
    def record_serveded_customer(self, info):
        served_list = []
        for vehicle_id, route_prob in enumerate(self.action_trajectory[-1]):
            served_customer = None
            target_node = info[0][vehicle_id][0]
            for i, prob_node_tuple in enumerate(route_prob):
                if prob_node_tuple[1] == target_node:
                    served_customer = [tup[1] for tup in route_prob[:i+1]]
                    served_list.append(served_customer)
                    break
            if served_customer == None:
                served_list.append([0])
        self.served_customer_trajectory.append(served_list)
    
    def set_total_reward(self, total_reward):
        self.total_reward = total_reward

    def set_baseline(self, total_reward):
        self.baseline = total_reward

    def memory(self):
        self.state_memory_buffer.append(self.state_trajectory)
        self.action_memory_buffer.append(self.action_trajectory)
        self.served_customer_memory_buffer.append(self.served_customer_trajectory)
        self.advantage_memory_buffer.append(self.total_reward-self.baseline)

        self.state_trajectory = []
        self.action_trajectory = []
        self.served_customer_trajectory = []
    
    def update_parameter(self):
        old_log_probs = torch.empty(self.batch_size, dtype=torch.float64, device=self.device)
        advantage = torch.Tensor(self.advantage_memory_buffer).to(self.device)
        for b in range(self.batch_size):
            action_trajectory = self.action_memory_buffer[b]
            served_customer_trajectory = self.served_customer_memory_buffer[b]                
            log_prob = self.get_old_log_prob(action_trajectory, served_customer_trajectory)
            old_log_probs[b] = log_prob
        old_log_probs = old_log_probs.detach()

        for _ in range(self.reuse_times):
            new_log_probs = torch.empty(self.batch_size, dtype=torch.float64, device=self.device)
            for b in range(self.batch_size):
                state_trajectory = self.state_memory_buffer[b]
                action_trajectory = self.action_memory_buffer[b]
                served_customer_trajectory = self.served_customer_memory_buffer[b]
                log_prob = self.get_new_log_prob(state_trajectory, action_trajectory, served_customer_trajectory)
                new_log_probs[b] = log_prob
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            loss = torch.mean(-torch.min(surr1, surr2))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.state_memory_buffer = []
        self.action_memory_buffer = []
        self.served_customer_memory_buffer = []
        self.advantage_memory_buffer = []

    def get_old_log_prob(self, action_trajectory, served_customer_trajectory):
        prob_list = []
        for t in range(len(action_trajectory)):
            action = action_trajectory[t]
            served_customer = served_customer_trajectory[t]

            for vehicle_id, route_prob in enumerate(action):
                served_num = len(served_customer[vehicle_id])
                prob = [tup[0] for tup in route_prob[:served_num]]
                prob_list.extend(prob)

        prob_tensor = torch.stack(prob_list)
        log_sum = torch.sum(torch.log(prob_tensor))
        return log_sum

    def get_new_log_prob(self, state_trajectory, action_trajectory, served_customer_trajectory):
        prob_list = []
        for t in range(len(state_trajectory)):
            state = state_trajectory[t]
            action = action_trajectory[t]
            served_customer = served_customer_trajectory[t]

            prob = self.model.prob_forward(state, action)
            for vehicle_id, route_prob in enumerate(prob):
                served_num = len(served_customer[vehicle_id])
                prob_list.extend(route_prob[:served_num])

        prob_tensor = torch.stack(prob_list)
        log_sum = torch.sum(torch.log(prob_tensor))
        return log_sum

    def save_model(self, epoch):
        self.model.save_model(epoch)