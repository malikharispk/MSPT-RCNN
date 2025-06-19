import torch
import torch.nn as nn
from .neighborhood_embed import NeighborhoodEmbedding
from .offset_attention import OffsetAttention
from .rpn import RPN

class MSPTRCNN(nn.Module):
    def __init__(self, num_classes=3, num_points=16384, num_neighbors=16):
        super().__init__()
        self.num_classes = num_classes
        self.num_points = num_points
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Neighborhood embedding layers
        self.neigh_embed1 = NeighborhoodEmbedding(128, 256, num_neighbors)
        self.neigh_embed2 = NeighborhoodEmbedding(256, 512, num_neighbors)
        
        # Offset attention layers
        self.offset_attn1 = OffsetAttention(256)
        self.offset_attn2 = OffsetAttention(512)
        
        # RPN
        self.rpn = RPN(512)
        
        # RoI pooling and head (simplified for this example)
        self.roi_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.cls_head = nn.Linear(128, num_classes)
        self.reg_head = nn.Linear(128, 7)
        
    def forward(self, xyz_features):
        """
        Args:
            xyz_features: (B, N, 4) input point cloud with XYZ and intensity
        Returns:
            rpn_cls: (B, N, num_anchors, num_classes) RPN classification
            rpn_reg: (B, N, num_anchors, 7) RPN regression
            roi_cls: (B, num_rois, num_classes) RoI classification
            roi_reg: (B, num_rois, 7) RoI regression
        """
        B, N, _ = xyz_features.shape
        xyz = xyz_features[:, :, :3]  # (B, N, 3)
        features = xyz_features[:, :, 3:]  # (B, N, 1)
        
        # Initial embedding
        features = self.input_embed(xyz_features.view(B*N, -1))  # (B*N, 128)
        features = features.view(B, N, -1)  # (B, N, 128)
        
        # Neighborhood embedding 1
        features = self.neigh_embed1(xyz, features)  # (B, N, 256)
        features = features.permute(0, 2, 1)  # (B, 256, N)
        
        # Offset attention 1
        features = self.offset_attn1(xyz, features)  # (B, 256, N)
        features = features.permute(0, 2, 1)  # (B, N, 256)
        
        # Neighborhood embedding 2
        features = self.neigh_embed2(xyz, features)  # (B, N, 512)
        features = features.permute(0, 2, 1)  # (B, 512, N)
        
        # Offset attention 2
        features = self.offset_attn2(xyz, features)  # (B, 512, N)
        
        # RPN
        rpn_cls, rpn_reg = self.rpn(features)
        
        # For simplicity, we'll skip the actual RoI pooling and just show the structure
        # In practice, you would use the proposals from RPN to pool features
        roi_features = features.mean(dim=2)  # (B, 512) - simplified
        roi_features = self.roi_head(roi_features)
        roi_cls = self.cls_head(roi_features)
        roi_reg = self.reg_head(roi_features)
        
        # Expand to match expected output shapes
        roi_cls = roi_cls.unsqueeze(1).expand(-1, 100, -1)  # (B, 100, num_classes)
        roi_reg = roi_reg.unsqueeze(1).expand(-1, 100, -1)  # (B, 100, 7)
        
        return {
            'rpn_cls': rpn_cls,
            'rpn_reg': rpn_reg,
            'roi_cls': roi_cls,
            'roi_reg': roi_reg
        }
