import torch
import torch.nn as nn
import torch.nn.functional as F

class NeighborhoodEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, num_neighbors=16):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, xyz, features):
        """
        Args:
            xyz: (B, N, 3) coordinates of points
            features: (B, N, C) features of points
        Returns:
            new_features: (B, N, C_out) enhanced features
        """
        B, N, _ = xyz.shape
        
        # Find k-nearest neighbors
        dist = torch.cdist(xyz, xyz)  # (B, N, N)
        _, knn_indices = torch.topk(dist, self.num_neighbors, dim=2, largest=False)  # (B, N, K)
        
        # Gather neighbor features
        knn_xyz = torch.gather(
            xyz.unsqueeze(2).expand(-1, -1, self.num_neighbors, -1),
            1,
            knn_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
        )  # (B, N, K, 3)
        
        knn_features = torch.gather(
            features.unsqueeze(2).expand(-1, -1, self.num_neighbors, -1),
            1,
            knn_indices.unsqueeze(-1).expand(-1, -1, -1, features.shape[-1])
        )  # (B, N, K, C)
        
        # Compute relative positions and features
        relative_xyz = knn_xyz - xyz.unsqueeze(2)  # (B, N, K, 3)
        relative_features = knn_features - features.unsqueeze(2)  # (B, N, K, C)
        
        # Concatenate and process
        combined = torch.cat([relative_xyz, relative_features], dim=-1)  # (B, N, K, 3+C)
        combined = combined.view(B * N * self.num_neighbors, -1)
        
        # Apply MLP
        new_features = self.mlp(combined)  # (B*N*K, C_out)
        new_features = new_features.view(B, N, self.num_neighbors, -1)
        
        # Max pooling over neighbors
        new_features = torch.max(new_features, dim=2)[0]  # (B, N, C_out)
        
        return new_features
