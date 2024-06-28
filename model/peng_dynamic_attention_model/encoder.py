import torch
import torch.nn as nn
import torch.nn.functional as F

class SubLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn_layer = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ff_1 = nn.Linear(embed_dim, ff_dim)
        self.ff_2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        attn_output = self.attn_layer(x, x, x)[0]
        tanh_1_output = F.tanh(torch.add(x, attn_output))
        ff_1_output = self.ff_1(tanh_1_output)
        ff_2_output = self.ff_2(F.relu(ff_1_output))
        tanh_2_output = F.tanh(torch.add(tanh_1_output, ff_2_output))

        return tanh_2_output

class Encoder(nn.Module):
    node_info = None

    def __init__(self, feature_dim, embed_dim, num_heads, num_layers, device):
        super().__init__()
        self.device = device
        self.ff_dim = embed_dim*4
        self.num_layers = num_layers

        self.feature_embed = nn.Linear(feature_dim, embed_dim)
        self.feature_layers = nn.ModuleList([SubLayer(embed_dim, num_heads, self.ff_dim) for _ in range(num_layers)])

    def set_node_info(self, node_info):
        self.node_info = node_info

    def update_feature(self, x):
        for i, info in enumerate(self.node_info):
            if info[1]:
                x[i, 2:] = 0
        return x
    
    def forward(self, x):   
        x = self.update_feature(x)
        x = x.to(self.device)
        x = self.feature_embed(x)
        for i in range(self.num_layers):
            x = self.feature_layers[i](x)
        
        return x
