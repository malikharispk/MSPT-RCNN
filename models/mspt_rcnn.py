import torch
import torch.nn as nn
from .embedding import NeighborhoodEmbeddingModule
from .offset_attention import OffsetAttentionLayer
from .jump_offset_attention import JumpConnectionOffsetAttentionModule
from .rpn import RegionProposalNetwork

class MSPTRCNN(nn.Module):
    def __init__(self):
        super(MSPTRCNN, self).__init__()
        # Define modules
        self.embedding = NeighborhoodEmbeddingModule()
        self.attention_module = JumpConnectionOffsetAttentionModule(input_dim=128, output_dim=256)
        self.rpn = RegionProposalNetwork(input_dim=256, output_dim=7)  # Example output for 3D box (x, y, z, h, w, l, orientation)

    def forward(self, x):
        x = self.embedding(x)  # Neighborhood embedding
        x = self.attention_module(x)  # Apply offset attention
        rpn_out = self.rpn(x)  # Generate initial bounding boxes
        return rpn_out
