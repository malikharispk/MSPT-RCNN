import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def visualize_point_cloud_with_boxes(points, boxes=None, title="Point Cloud with 3D Boxes"):
    """
    Visualize point cloud with optional 3D bounding boxes
    Args:
        points: (N, 3) point cloud
        boxes: (M, 7) 3D boxes [x, y, z, l, w, h, ry]
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=1, alpha=0.5)
    
    # Plot boxes if provided
    if boxes is not None:
        from .bbox_utils import boxes_to_corners_3d
        corners = boxes_to_corners_3d(torch.from_numpy(boxes)).numpy()
        
        for box_corners in corners:
            # Define the 6 faces of the box
            faces = [
                [box_corners[0], box_corners[1], box_corners[2], box_corners[3]],  # bottom
                [box_corners[4], box_corners[5], box_corners[6], box_corners[7]],  # top
                [box_corners[0], box_corners[1], box_corners[5], box_corners[4]],  # front
                [box_corners[2], box_corners[3], box_corners[7], box_corners[6]],  # back
                [box_corners[1], box_corners[2], box_corners[6], box_corners[5]],  # right
                [box_corners[0], box_corners[3], box_corners[7], box_corners[4]]   # left
            ]
            
            # Add each face with transparency
            face_collection = Poly3DCollection(
                faces, 
                linewidths=1, 
                edgecolors='r',
                facecolors=(1, 0, 0, 0.1)  # Red with transparency
            )
            ax.add_collection3d(face_collection)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()
