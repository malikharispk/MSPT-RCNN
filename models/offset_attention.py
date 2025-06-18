import torch
import torch.nn as nn

class OffsetAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(OffsetAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)  # Self-attention
        return self.linear(attn_output)
