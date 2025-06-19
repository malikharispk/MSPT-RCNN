import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import numpy as np

from models.mspt_rcnn import MSPTRCNN
from data.kitti_dataset import KITTIDataset
from utils.metrics import calculate_map
from utils.visualizer import visualize_point_cloud_with_boxes

def test_model(config_path, model_path, sample_idx=0):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build model
    model = MSPTRCNN(
        num_classes=config['model']['num_classes'],
        num_points=config['model']['num_points'],
        num_neighbors=config['model']['num_neighbors']
    ).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Build dataloader
    dataset = KITTIDataset(
        root_dir=config['data']['root_dir'],
        split='testing',
        num_points=config['model']['num_points']
    )
    
    # Get a sample
    sample = dataset[sample_idx]
    point_cloud = sample['point_cloud'].unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(point_cloud)
    
    # Get predictions
    _, pred_labels = torch.max(outputs['roi_cls'], dim=2)
    pred_scores = torch.softmax(outputs['roi_cls'], dim=2).max(dim=2)[0]
    pred_boxes = outputs['roi_reg']
    
    # Filter predictions by score
    score_threshold = 0.5
    mask = pred_scores[0] > score_threshold
    filtered_boxes = pred_boxes[0][mask].cpu().numpy()
    filtered_labels = pred_labels[0][mask].cpu().numpy()
    filtered_scores = pred_scores[0][mask].cpu().numpy()
    
    # Visualize
    print(f"Sample {sample_idx} - Found {len(filtered_boxes)} objects")
    for i, (box, label, score) in enumerate(zip(filtered_boxes, filtered_labels, filtered_scores)):
        print(f"Object {i}: Class {label}, Score {score:.2f}, Box {box}")
    
    visualize_point_cloud_with_boxes(
        point_cloud[0, :, :3].cpu().numpy(),
        filtered_boxes
    )

if __name__ == "__main__":
    test_model('configs/config.yaml', 'checkpoints/best_model.pth')
