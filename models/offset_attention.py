import torch
import torch.nn as nn
import torch.nn.functional as F

class OffsetAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels, 1)
        self.k_conv = nn.Conv1d(channels, channels, 1)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.pos_conv = nn.Conv1d(3, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, xyz, features):
        """
        Args:
            xyz: (B, N, 3) coordinates of points
            features: (B, C, N) features of points
        Returns:
            new_features: (B, C, N) enhanced features
        """
        B, C, N = features.shape
        
        # Compute queries, keys, values
        q = self.q_conv(features).permute(0, 2, 1)  # (B, N, C)
        k = self.k_conv(features)  # (B, C, N)
        v = self.v_conv(features)  # (B, C, N)
        
        # Compute position encoding
        pos_enc = self.pos_conv(xyz.permute(0, 2, 1))  # (B, C, N)
        
        # Compute attention weights
        attn = torch.bmm(q, k) / (C ** 0.5)  # (B, N, N)
        attn = self.softmax(attn)
        
        # Apply attention to values and position encoding
        attended_values = torch.bmm(v, attn.permute(0, 2, 1))  # (B, C, N)
        attended_pos = torch.bmm(pos_enc, attn.permute(0, 2, 1))  # (B, C, N)
        
        # Combine and apply residual connection
        new_features = attended_values + attended_pos + features  # (B, C, N)
        new_features = self.after_norm(new_features)
        new_features = self.act(new_features)
        
        return new_features
