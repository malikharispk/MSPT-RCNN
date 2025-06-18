import torch
from torch.utils.data import DataLoader
from data.kitti_dataset import KITTIDataset
from models.mspt_rcnn import MSPTRCNN
import torch.optim as optim
import torch.nn.functional as F

# Load dataset
dataset = KITTIDataset(root_dir='/path/to/kitti')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model and optimizer
model = MSPTRCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    model.train()
    running_loss = 0.0
    for data in dataloader:
        optimizer.zero_grad()
        output = model(data)
        # Loss function: Placeholder
        loss = F.mse_loss(output, data)  # Replace with proper loss function
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")
