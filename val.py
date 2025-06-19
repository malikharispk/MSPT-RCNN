import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from models.mspt_rcnn import MSPTRCNN
from data.kitti_dataset import KITTIDataset
from utils.metrics import calculate_map

def validate_model(config_path, model_path):
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
        split='validation',  # Assuming you have a validation split
        num_points=config['model']['num_points']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # Loss functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss()
    
    # Run validation
    val_loss, mAP = validate(
        model, dataloader, criterion_cls, criterion_reg, device, config
    )
    
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"mAP: {mAP:.4f}")

if __name__ == "__main__":
    validate_model('configs/config.yaml', 'checkpoints/best_model.pth')
