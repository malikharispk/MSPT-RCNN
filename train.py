import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import os

from models.mspt_rcnn import MSPTRCNN
from data.kitti_dataset import KITTIDataset
from utils.metrics import calculate_map
from utils.visualizer import visualize_point_cloud_with_boxes

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def build_model(config):
    model = MSPTRCNN(
        num_classes=config['model']['num_classes'],
        num_points=config['model']['num_points'],
        num_neighbors=config['model']['num_neighbors']
    )
    return model

def build_dataloader(config):
    dataset = KITTIDataset(
        root_dir=config['data']['root_dir'],
        split=config['data']['split'],
        num_points=config['model']['num_points']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=config['data']['shuffle'],
        num_workers=config['data']['num_workers']
    )
    return dataloader

def train_one_epoch(model, dataloader, optimizer, criterion_cls, criterion_reg, device):
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move data to device
        point_cloud = batch['point_cloud'].to(device)
        gt_boxes = batch['labels']['boxes'].to(device)
        gt_classes = batch['labels']['classes'].to(device)
        
        # Forward pass
        outputs = model(point_cloud)
        
        # Compute losses
        # RPN classification loss
        rpn_cls_loss = criterion_cls(
            outputs['rpn_cls'].view(-1, outputs['rpn_cls'].shape[-1]),
            gt_classes.view(-1)
        )
        
        # RPN regression loss (only for positive anchors)
        pos_mask = (gt_classes > 0).float()
        rpn_reg_loss = criterion_reg(
            outputs['rpn_reg'] * pos_mask.unsqueeze(-1),
            gt_boxes.unsqueeze(2) * pos_mask.unsqueeze(-1)
        )
        
        # RoI classification loss
        roi_cls_loss = criterion_cls(
            outputs['roi_cls'].view(-1, outputs['roi_cls'].shape[-1]),
            gt_classes.view(-1)
        )
        
        # RoI regression loss
        roi_reg_loss = criterion_reg(
            outputs['roi_reg'],
            gt_boxes.unsqueeze(1)
        )
        
        # Total loss
        loss = rpn_cls_loss + rpn_reg_loss + roi_cls_loss + roi_reg_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion_cls, criterion_reg, device, config):
    model.eval()
    total_loss = 0.0
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move data to device
            point_cloud = batch['point_cloud'].to(device)
            gt_boxes = batch['labels']['boxes'].to(device)
            gt_classes = batch['labels']['classes'].to(device)
            
            # Forward pass
            outputs = model(point_cloud)
            
            # Compute losses
            rpn_cls_loss = criterion_cls(
                outputs['rpn_cls'].view(-1, outputs['rpn_cls'].shape[-1]),
                gt_classes.view(-1)
            )
            
            pos_mask = (gt_classes > 0).float()
            rpn_reg_loss = criterion_reg(
                outputs['rpn_reg'] * pos_mask.unsqueeze(-1),
                gt_boxes.unsqueeze(2) * pos_mask.unsqueeze(-1)
            )
            
            roi_cls_loss = criterion_cls(
                outputs['roi_cls'].view(-1, outputs['roi_cls'].shape[-1]),
                gt_classes.view(-1)
            )
            
            roi_reg_loss = criterion_reg(
                outputs['roi_reg'],
                gt_boxes.unsqueeze(1)
            )
            
            loss = rpn_cls_loss + rpn_reg_loss + roi_cls_loss + roi_reg_loss
            total_loss += loss.item()
            
            # Collect predictions for mAP calculation
            _, pred_labels = torch.max(outputs['roi_cls'], dim=2)
            pred_scores = torch.softmax(outputs['roi_cls'], dim=2).max(dim=2)[0]
            
            all_pred_boxes.append(outputs['roi_reg'].cpu())
            all_pred_scores.append(pred_scores.cpu())
            all_pred_labels.append(pred_labels.cpu())
            all_gt_boxes.append(gt_boxes.cpu())
            all_gt_labels.append(gt_classes.cpu())
    
    # Calculate mAP
    mAP = calculate_map(
        all_pred_boxes, all_pred_scores, all_pred_labels,
        all_gt_boxes, all_gt_labels,
        iou_threshold=config['eval']['iou_threshold']
    )
    
    return total_loss / len(dataloader), mAP

def main():
    # Load config
    config = load_config('configs/config.yaml')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build model
    model = build_model(config).to(device)
    
    # Build dataloader
    dataloader = build_dataloader(config)
    
    # Setup optimizer and loss functions
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['train']['lr_decay_step'],
        gamma=config['train']['lr_decay_rate']
    )
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss()
    
    # Training loop
    best_mAP = 0.0
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(config['train']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['train']['epochs']}")
        
        # Train
        train_loss = train_one_epoch(
            model, dataloader, optimizer, criterion_cls, criterion_reg, device
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, mAP = validate(
            model, dataloader, criterion_cls, criterion_reg, device, config
        )
        print(f"Val Loss: {val_loss:.4f}, mAP: {mAP:.4f}")
        
        # Save best model
        if mAP > best_mAP:
            best_mAP = mAP
            torch.save(model.state_dict(), f"checkpoints/best_model.pth")
            print(f"Saved best model with mAP: {best_mAP:.4f}")
        
        # Update learning rate
        scheduler.step()
    
    print("Training completed!")

if __name__ == "__main__":
    main()
