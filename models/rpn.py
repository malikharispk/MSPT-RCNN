import torch
import torch.nn as nn
import torch.nn.functional as F

class RPN(nn.Module):
    def __init__(self, in_channels, num_anchors=2, num_classes=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        # Shared convolutional layers
        self.conv1 = nn.Conv1d(in_channels, in_channels, 1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        
        self.conv2 = nn.Conv1d(in_channels, in_channels, 1)
        self.bn2 = nn.BatchNorm1d(in_channels)
        
        # Classification head
        self.cls_conv = nn.Conv1d(in_channels, num_anchors * num_classes, 1)
        
        # Regression head
        self.reg_conv = nn.Conv1d(in_channels, num_anchors * 7, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Args:
            x: (B, C, N) input features
        Returns:
            cls_preds: (B, N, num_anchors, num_classes) classification scores
            reg_preds: (B, N, num_anchors, 7) box predictions
        """
        B, C, N = x.shape
        
        # Shared features
        shared = self.relu(self.bn1(self.conv1(x)))
        shared = self.relu(self.bn2(self.conv2(shared)))
        
        # Classification predictions
        cls_preds = self.cls_conv(shared)  # (B, num_anchors*num_classes, N)
        cls_preds = cls_preds.view(B, self.num_anchors, self.num_classes, N)
        cls_preds = cls_preds.permute(0, 3, 1, 2)  # (B, N, num_anchors, num_classes)
        
        # Regression predictions
        reg_preds = self.reg_conv(shared)  # (B, num_anchors*7, N)
        reg_preds = reg_preds.view(B, self.num_anchors, 7, N)
        reg_preds = reg_preds.permute(0, 3, 1, 2)  # (B, N, num_anchors, 7)
        
        return cls_preds, reg_preds
