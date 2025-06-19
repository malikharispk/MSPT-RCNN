import torch
from kitti_dataset import KITTIDataset
from torch.utils.data import DataLoader
from models.mspt_rcnn import MSPTRCNN
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
config = {
    'batch_size': 4
}

# Initialize validation dataset and dataloaders
val_dataset = KITTIDataset(data_dir='/path/to/kitti', split='testing')
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

# Load model and set it to evaluation mode
model = MSPTRCNN(in_channels=6, rpn_out_channels=64, classifier_hidden_channels=128, classifier_out_channels=6).cuda()
model.load_state_dict(torch.load('checkpoints/mspt_rcnn_epoch_20.pth'))
model.eval()

# Evaluation loop
with torch.no_grad():
    for data in val_loader:
        inputs, labels = data.x.cuda(), data.y.cuda()
        outputs = model(inputs)
        
        # Calculate metrics (IoU, mAP, etc.)
        # You can implement additional evaluation code here

print("Validation complete")
