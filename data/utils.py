import numpy as np
import torch

def rotate_points_along_z(points, angle):
    """
    Rotate points along z-axis
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle in radians
    Returns:
        rotated_points: (B, N, 3 + C)
    """
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot

def boxes_to_corners_3d(boxes):
    """
    Convert 3D bounding boxes to corners
    Args:
        boxes: (N, 7) [x, y, z, l, w, h, ry]
    Returns:
        corners_3d: (N, 8, 3)
    """
    template = torch.tensor([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]
    ], dtype=torch.float32, device=boxes.device) / 2
    
    corners = boxes[:, None, 3:6] * template[None, :, :]
    
    # Rotate around y-axis
    rot_sin = torch.sin(boxes[:, 6])
    rot_cos = torch.cos(boxes[:, 6])
    rot_mat = torch.stack([
        rot_cos, torch.zeros_like(rot_cos), rot_sin,
        torch.zeros_like(rot_cos), torch.ones_like(rot_cos), torch.zeros_like(rot_cos),
        -rot_sin, torch.zeros_like(rot_cos), rot_cos
    ], dim=1).view(-1, 3, 3)
    
    corners = torch.matmul(corners, rot_mat.transpose(1, 2))
    
    # Translate to center
    corners += boxes[:, None, 0:3]
    
    return corners

def mask_points_in_boxes(points, boxes):
    """
    Mask points that are inside boxes
    Args:
        points: (N, 3)
        boxes: (M, 7)
    Returns:
        point_masks: (N, M) bool
    """
    corners = boxes_to_corners_3d(boxes)  # (M, 8, 3)
    
    # Compute axis-aligned bounding boxes for each box
    min_bounds = corners.min(dim=1)[0]  # (M, 3)
    max_bounds = corners.max(dim=1)[0]  # (M, 3)
    
    # Expand dimensions for broadcasting
    points_exp = points.unsqueeze(1)  # (N, 1, 3)
    min_exp = min_bounds.unsqueeze(0)  # (1, M, 3)
    max_exp = max_bounds.unsqueeze(0)  # (1, M, 3)
    
    # Check if points are within AABBs
    in_aabb = torch.all((points_exp >= min_exp) & (points_exp <= max_exp), dim=2)
    
    return in_aabb
