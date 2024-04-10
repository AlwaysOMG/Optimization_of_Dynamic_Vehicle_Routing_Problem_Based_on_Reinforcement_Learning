import configparser
import torch.nn as nn

from model.new_dynamic_attention_model.encoder import Encoder
from model.new_dynamic_attention_model.decoder import Decoder

config = configparser.ConfigParser()
config.read("./model/new_dynamic_attention_model/model.cfg")
parameter_config = config['parameter']

class DynamicAttentionModel(nn.Module):
    embed_dim = int(parameter_config["embed_dim"])
    num_heads = int(parameter_config["num_heads"])
    num_layers = int(parameter_config["num_layers"])
    clip_c = int(parameter_config["clip_c"])

    def __init__(self, feature_dim):
        super().__init__()
        static_feature_dim = feature_dim[0]
        dynamic_feature_dim = feature_dim[1]
        self.encoder = Encoder(static_feature_dim, dynamic_feature_dim, 
                               self.embed_dim, self.num_heads, self.num_layers)
        self.decoder = Decoder(self.embed_dim, self.num_heads, self.clip_c)
    
    def forward(self, x):
        encoder_output = self.encoder(x)
        route = self.decoder(encoder_output)

        return route
    
    def set_info(self, info):
        self.decoder.set_vehicle_info(info[0])
        self.decoder.set_node_info(info[1])