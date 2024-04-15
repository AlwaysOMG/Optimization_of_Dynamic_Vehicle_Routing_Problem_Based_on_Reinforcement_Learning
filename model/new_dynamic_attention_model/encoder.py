import torch
import torch.nn as nn
import torch.nn.functional as F

class Sublayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn_layer = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ff_1 = nn.Linear(embed_dim, ff_dim)
        self.ff_2 = nn.Linear(ff_dim, embed_dim)
    
    def forward(self, x):
        attn_output = self.attn_layer(x, x, x)[0]
        tanh_1_output = torch.tanh(torch.add(x, attn_output))
        ff_1_output = self.ff_1(tanh_1_output)
        ff_2_output = self.ff_2(F.relu(ff_1_output))
        tanh_2_output = torch.tanh(torch.add(tanh_1_output, ff_2_output))

        return tanh_2_output

class Encoder(nn.Module):
    def __init__(self, static_feature_dim, dynamic_feature_dim, 
                 embed_dim, num_heads, num_layers, device):
        super().__init__()
        self.device = device
        self.ff_dim = embed_dim*4
        self.num_layers = num_layers

        self.static_norm = nn.LayerNorm(static_feature_dim)
        self.static_embed = nn.Linear(static_feature_dim, embed_dim)
        self.static_layers = nn.ModuleList([Sublayer(embed_dim, num_heads, self.ff_dim) for _ in range(num_layers)])
        
        self.dynamic_norm = nn.LayerNorm(dynamic_feature_dim)
        self.dynamic_embed = nn.Linear(dynamic_feature_dim, embed_dim)
        self.dynamic_layers = nn.ModuleList([Sublayer(embed_dim, num_heads, self.ff_dim) for _ in range(num_layers)])

        self.combined_layer = nn.Linear(embed_dim*2, embed_dim)
    
    def forward(self, x):
        static_x = x[0].to(self.device)
        static_x = self.static_norm(static_x)
        static_x = self.static_embed(static_x)
        for i in range(self.num_layers):
            static_x = self.static_layers[i](static_x)
        
        dynamic_x = x[1].to(self.device)
        dynamic_x = self.dynamic_norm(dynamic_x)
        dynamic_x = self.dynamic_embed(dynamic_x)
        for i in range(self.num_layers):
            dynamic_x = self.dynamic_layers[i](dynamic_x)

        combined_x = self.combined_layer(torch.cat((static_x, dynamic_x), dim=1))
        
        return combined_x
