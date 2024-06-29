import configparser
import torch
import torch.nn as nn

from model.new_dynamic_attention_model.encoder import Encoder
from model.new_dynamic_attention_model.decoder import Decoder

class DynamicAttentionModel(nn.Module):
    device = torch.device("cuda:0")

    config = configparser.ConfigParser()
    config.read("./config.cfg")
    embed_dim = int(config['parameter']["embed_dim"])
    num_heads = int(config['parameter']["num_heads"])
    num_layers = int(config['parameter']["num_layers"])
    clip_c = int(config['parameter']["clip_c"])

    customer_num = int(config["instance"]["customer_num"])
    static_feature_dim = int(config["instance"]["feature_dim"])
    dynamic_feature_dim = customer_num+1

    def __init__(self):
        super().__init__()
        self.encoder = Encoder(self.static_feature_dim, self.dynamic_feature_dim, 
                               self.embed_dim, self.num_heads, self.num_layers, 
                               self.device)
        self.decoder = Decoder(self.embed_dim, self.num_heads, self.clip_c, 
                               self.device)
        
        self.to(self.device)
    
    def forward(self, x, info, is_greedy):
        self.decoder.set_vehicle_info(info[0])
        self.decoder.set_node_info(info[1])
        self.decoder.set_decision_time(info[2])
        self.decoder.set_travel_time(x[1])

        encoder_output = self.encoder(x)
        route, prob = self.decoder(encoder_output, is_greedy)

        return route, prob
    
    def save_model(self, epoch):
        torch.save(self.state_dict(), f"./model/new_dynamic_attention_model/parameter/{epoch+1}.pth")

    def prob_forward(self, state, action):
        """
        Output the prob of specific action
        """
        self.decoder.set_vehicle_info(state[1][0])
        self.decoder.set_node_info(state[1][1])

        encoder_output = self.encoder(state[0])
        prob = self.decoder.get_route_prob(encoder_output, action)

        return prob