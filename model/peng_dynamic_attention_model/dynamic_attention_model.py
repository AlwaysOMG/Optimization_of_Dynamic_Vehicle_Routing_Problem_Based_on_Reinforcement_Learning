import configparser
import copy
import torch
import torch.nn as nn

from model.peng_dynamic_attention_model.encoder import Encoder
from model.peng_dynamic_attention_model.decoder import Decoder

class DynamicAttentionModel(nn.Module):
    device = torch.device("cuda:0")

    config = configparser.ConfigParser()
    config.read("./config.cfg")
    parameter_config = config['parameter']
    embed_dim = int(parameter_config["embed_dim"])
    num_heads = int(parameter_config["num_heads"])
    num_layers = int(parameter_config["num_layers"])
    clip_c = int(parameter_config["clip_c"])

    customer_num = int(config["instance"]["customer_num"])
    feature_dim = int(config["instance"]["feature_dim"])

    def __init__(self):
        super().__init__()
        self.encoder = Encoder(self.feature_dim, self.embed_dim, self.num_heads, self.num_layers, 
                               self.device)
        self.decoder = Decoder(self.embed_dim, self.num_heads, self.clip_c, 
                               self.device)
        
        self.to(self.device)
    
    def forward(self, x, info, is_greedy):
        vehicle_info = info[0]
        node_info = info[1]
        route_list = []
        prob_list = []
        for v_info in vehicle_info:
            self.encoder.set_node_info(node_info)
            encoder_output = self.encoder(x[0])

            self.decoder.set_vehicle_info(v_info)
            self.decoder.set_node_info(node_info)
            route, prob = self.decoder(encoder_output, is_greedy)

            node_info = self.update_node_info(node_info, route)
            route_list.append(route)
            prob_list.append(prob)

        return route_list, prob_list
    
    def update_node_info(self, node_info, route):
        update_info = copy.deepcopy(node_info)
        for node in route:
            if node != 0:
                update_info[node][1] = True
        return update_info

    def save_model(self, epoch):
        torch.save(self.state_dict(), f"./model/peng_dynamic_attention_model/parameter/{epoch+1+12}.pth")
