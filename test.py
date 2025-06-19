import torch
from kitti_dataset import KITTIDataset
from torch.utils.data import DataLoader
from models.mspt_rcnn import MSPTRCNN

# Initialize the test dataset and dataloaders
test_dataset = KITTIDataset(data_dir='/path/to/kitti', split='testing')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the trained model
model = MSPTRCNN(in_channels=6, rpn_out_channels=64, classifier_hidden_channels=128, classifier_out_channels=6).cuda()
model.load_state_dict(torch.load('checkpoints/mspt_rcnn_epoch_20.pth'))
model.eval()

# Testing loop
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data.x.cuda(), data.y.cuda()
        outputs = model(inputs)
        
        # Visualize the results (e.g., plot the 3D bounding boxes)
        # You can add visualization code here
        print(outputs)

print("Testing complete")
