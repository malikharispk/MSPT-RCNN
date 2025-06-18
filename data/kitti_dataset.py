import os
import torch
from torch.utils.data import Dataset
import numpy as np

class KITTIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the point cloud data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.bin')]  # Example of loading .bin files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Loading the .bin file, assuming LiDAR data with 4 columns (x, y, z, intensity)
        file_path = os.path.join(self.root_dir, self.files[idx])
        data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

        if self.transform:
            data = self.transform(data)

        # Return data as tensor
        return torch.tensor(data)
