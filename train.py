import torch
from torch.utils.data import DataLoader
from kitti_dataset import KITTIDataset
from models.mspt_rcnn import MSPTRCNN
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import os

# Hyperparameters
config = {
    'batch_size': 4,
    'epochs': 20,
    'lr': 1e-4,
    'rpn_out_channels': 64,
    'classifier_hidden_channels': 128,
    'classifier_out_channels': 6
}

# Initialize dataset and dataloaders
train_dataset = KITTIDataset(data_dir='/path/to/kitti', split='training')
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

# Model, loss, optimizer
model = MSPTRCNN(in_channels=6, 
                 rpn_out_channels=config['rpn_out_channels'], 
                 classifier_hidden_channels=config['classifier_hidden_channels'], 
                 classifier_out_channels=config['classifier_out_channels']).cuda()

optimizer = Adam(model.parameters(), lr=config['lr'])
criterion = nn.MSELoss()  # You can change this based on your task (e.g., classification loss)

# Setup TensorBoard
writer = SummaryWriter(log_dir='./logs')

# Training loop
for epoch in range(config['epochs']):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data.x.cuda(), data.y.cuda()
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % 10 == 9:  # Print every 10 batches
            print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/10}")
            writer.add_scalar('Training Loss', running_loss/10, epoch * len(train_loader) + i)
            running_loss = 0.0
    
    # Save model checkpoint after every epoch
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    torch.save(model.state_dict(), f'checkpoints/mspt_rcnn_epoch_{epoch+1}.pth')

print("Finished Training")
