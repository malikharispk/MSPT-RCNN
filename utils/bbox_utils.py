import torch

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

def transform_boxes(boxes, transform_mat):
    """
    Transform 3D boxes using a 4x4 transformation matrix
    Args:
        boxes: (N, 7) [x, y, z, l, w, h, ry]
        transform_mat: (4, 4) transformation matrix
    Returns:
        transformed_boxes: (N, 7) transformed boxes
    """
    corners = boxes_to_corners_3d(boxes)  # (N, 8, 3)
    
    # Convert to homogeneous coordinates
    corners_homo = torch.cat([
        corners, 
        torch.ones_like(corners[..., :1], device=corners.device)
    ], dim=-1)  # (N, 8, 4)
    
    # Transform corners
    transformed_corners = torch.matmul(corners_homo, transform_mat.t())  # (N, 8, 4)
    transformed_corners = transformed_corners[..., :3]  # (N, 8, 3)
    
    # Compute new box parameters from transformed corners
    min_vals = transformed_corners.min(dim=1)[0]  # (N, 3)
    max_vals = transformed_corners.max(dim=1)[0]  # (N, 3)
    
    # Center
    center = (min_vals + max_vals) / 2
    
    # Dimensions
    dimensions = max_vals - min_vals
    
    # Rotation (simplified - in practice need more complex calculation)
    rotation_y = boxes[:, 6]  # Keep same rotation for simplicity
    
    transformed_boxes = torch.cat([
        center, 
        dimensions, 
        rotation_y.unsqueeze(-1)
    ], dim=-1)
    
    return transformed_boxes
