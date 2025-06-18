import torch
from torch.utils.data import DataLoader
from data.kitti_dataset import KITTIDataset
from models.mspt_rcnn import MSPTRCNN
from utils.metrics import iou, mAP

# Load dataset
dataset = KITTIDataset(root_dir='/path/to/kitti')
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Initialize model
model = MSPTRCNN()

# Evaluation loop
model.eval()
with torch.no_grad():
    for data in dataloader:
        output = model(data)
        # Compute evaluation metrics (IoU, mAP, etc.)
        iou_score = iou(output, data)
        map_score = mAP(output, data)
        print(f"IoU: {iou_score}, mAP: {map_score}")
