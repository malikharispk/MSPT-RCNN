import torch
import torch.nn as nn

class NeighborhoodEmbeddingModule(nn.Module):
    def __init__(self):
        super(NeighborhoodEmbeddingModule, self).__init__()
        self.lbr = nn.Sequential(
            nn.Linear(3, 64),  # Input 3D coordinates (x, y, z)
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.sg = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.lbr(x)
        x = self.sg(x)
        return x
