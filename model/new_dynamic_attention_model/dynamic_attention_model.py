import configparser
import torch
import torch.nn as nn

from model.new_dynamic_attention_model.encoder import Encoder
from model.new_dynamic_attention_model.decoder import Decoder

config = configparser.ConfigParser()
config.read("./config.cfg")
parameter_config = config['parameter']

class DynamicAttentionModel(nn.Module):
    device = torch.device("cuda:0")
    embed_dim = int(parameter_config["embed_dim"])
    num_heads = int(parameter_config["num_heads"])
    num_layers = int(parameter_config["num_layers"])
    clip_c = int(parameter_config["clip_c"])

    def __init__(self, customer_num, feature_dim):
        super().__init__()
        self.customer_num = customer_num
        static_feature_dim = feature_dim[0]
        dynamic_feature_dim = feature_dim[1]
        self.encoder = Encoder(static_feature_dim, dynamic_feature_dim, 
                               self.embed_dim, self.num_heads, self.num_layers, 
                               self.device)
        self.decoder = Decoder(self.embed_dim, self.num_heads, self.clip_c, 
                               self.device)
        
        self.to(self.device)
    
    def forward(self, x, info, is_greedy):
        self.decoder.set_vehicle_info(info[0])
        self.decoder.set_node_info(info[1])

        encoder_output = self.encoder(x)
        route, prob = self.decoder(encoder_output, is_greedy)

        return route, prob
    
    def save_model(self, epoch):
        torch.save(self.state_dict(), f"./model/new_dynamic_attention_model/parameter/{self.customer_num}/{self.embed_dim}_{self.num_layers}_{self.num_heads}_{epoch+1}.pth")
