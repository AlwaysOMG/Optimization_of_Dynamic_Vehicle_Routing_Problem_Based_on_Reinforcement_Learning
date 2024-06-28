import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ProbLayer(nn.Module):
    def __init__(self, embed_dim, clip_c):
        super().__init__()
        self.q_weight = nn.Linear(embed_dim, embed_dim)
        self.k_weight = nn.Linear(embed_dim, embed_dim)
        self.clip_c = clip_c
    
    def forward(self, x, mask):
        q = self.q_weight(x[0])
        k = self.k_weight(x[1])
        u = self.clip_c * F.tanh(torch.matmul(q, k.T))
        mask_u = torch.where(mask, float('-inf'), u)
        prob = F.softmax(mask_u, dim=-1)

        return prob
        
class Decoder(nn.Module):
    vehicle_info = None
    node_info = None
    travel_time_matrix = None

    def __init__(self, embed_dim, num_heads, clip_c, device):
        super().__init__()
        self.combined_layer = nn.Linear(embed_dim*2+1, embed_dim)
        self.attn_layer = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.prob_layer = ProbLayer(embed_dim, clip_c)
        self.device = device
    
    def set_vehicle_info(self, vehicle_info):
        self.vehicle_info = copy.deepcopy(vehicle_info)

    def set_node_info(self, node_info):
        self.node_info = copy.deepcopy(node_info)
        target_node_id = self.vehicle_info[0]
        if target_node_id != 0:
            self.node_info[target_node_id][1] = True

    def set_travel_time(self, travel_time):
        self.travel_time_matrix = travel_time

    def make_mean_mask(self):
        # mask the served customer
        return torch.tensor([node[1] == True for node in self.node_info])

    def make_vehicle_embed(self, x):
        mask = self.make_mean_mask()
        with torch.no_grad():
            mean_embed = x[~mask].mean(dim=0).unsqueeze(0)
            selected_node_embed = x[self.vehicle_info[0]].unsqueeze(0)
            capacity_tensor = torch.tensor([[self.vehicle_info[1]]], dtype=torch.float32).to(self.device)
        combined_embed = self.combined_layer(torch.cat((mean_embed, selected_node_embed, capacity_tensor), dim=1))

        return combined_embed
    
    def make_attn_mask(self):
        # mask the served customer and the customer over capacity
        check_list = [(node[1] == True) or (node[0] > self.vehicle_info[1])
                      for node in self.node_info]
        
        # mask the depot if the vehicle is at the depot
        if self.vehicle_info[0] == 0:
            check_list[0] = True
        
        #　if all nodes are mask, then unmask depot
        if all(check_list):
            check_list[0] = False

        return torch.tensor(check_list).to(self.device)

    def sample_customer(self, prob, is_greedy):
        id = torch.argmax(prob, dim=1).item() if is_greedy \
            else Categorical(prob).sample().item()

        return id
    
    def update_info(self, node_id):        
        self.vehicle_info[0] = node_id
        self.vehicle_info[1] -= self.node_info[node_id][0]

        if node_id != 0:
            self.node_info[node_id][1] = True
    
    def forward(self, x, is_greedy=False):
        route = []
        route_prob = []
        while True:
            vehicle_embed = self.make_vehicle_embed(x)
            attn_mask = self.make_attn_mask()
            context_vector = self.attn_layer(vehicle_embed, x, x, 
                                             key_padding_mask=attn_mask)[0]
                
            prob = self.prob_layer([context_vector, x], attn_mask)
            node_id = self.sample_customer(prob, is_greedy)              
            route.append(node_id)
            route_prob.append((prob[0][node_id], node_id))
                
            if node_id == 0:
                return route, route_prob
                
            self.update_info(node_id)
