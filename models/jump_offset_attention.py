import torch
import torch.nn as nn
from .offset_attention import OffsetAttentionLayer

class JumpConnectionOffsetAttentionModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(JumpConnectionOffsetAttentionModule, self).__init__()
        self.att1 = OffsetAttentionLayer(input_dim, output_dim)
        self.att2 = OffsetAttentionLayer(input_dim, output_dim)
        self.att3 = OffsetAttentionLayer(input_dim, output_dim)
        self.att4 = OffsetAttentionLayer(input_dim, output_dim)

    def forward(self, x):
        att1_out = self.att1(x)
        att2_out = self.att2(att1_out)
        att3_out = self.att3(att2_out)
        att4_out = self.att4(att3_out)
        return att4_out
