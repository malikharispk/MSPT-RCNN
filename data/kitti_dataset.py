import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pyquaternion import Quaternion

class KITTIDataset(Dataset):
    def __init__(self, root_dir, split='training', num_points=16384):
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.num_points = num_points
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Load sample indices from the dataset directory"""
        velodyne_dir = os.path.join(self.root_dir, 'velodyne')
        label_dir = os.path.join(self.root_dir, 'label_2')
        calib_dir = os.path.join(self.root_dir, 'calib')
        
        sample_ids = [f.split('.')[0] for f in os.listdir(velodyne_dir) if f.endswith('.bin')]
        samples = []
        
        for sample_id in sample_ids:
            sample = {
                'id': sample_id,
                'point_cloud_path': os.path.join(velodyne_dir, f'{sample_id}.bin'),
                'label_path': os.path.join(label_dir, f'{sample_id}.txt'),
                'calib_path': os.path.join(calib_dir, f'{sample_id}.txt')
            }
            samples.append(sample)
            
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load point cloud
        point_cloud = np.fromfile(sample['point_cloud_path'], dtype=np.float32).reshape(-1, 4)
        
        # Downsample if necessary
        if point_cloud.shape[0] > self.num_points:
            indices = np.random.choice(point_cloud.shape[0], self.num_points, replace=False)
            point_cloud = point_cloud[indices]
        
        # Load calibration
        calib = self._read_calibration(sample['calib_path'])
        
        # Load labels
        labels = self._read_labels(sample['label_path'], calib)
        
        # Convert to torch tensors
        point_cloud = torch.from_numpy(point_cloud)
        labels = {k: torch.from_numpy(v) for k, v in labels.items()}
        
        return {
            'point_cloud': point_cloud,
            'labels': labels,
            'calib': calib,
            'sample_id': sample['id']
        }
    
    def _read_calibration(self, calib_path):
        """Read calibration file and return as dict"""
        calib = {}
        with open(calib_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    calib[key.strip()] = np.array([float(x) for x in value.strip().split()])
        return calib
    
    def _read_labels(self, label_path, calib):
        """Read and parse label file"""
        boxes = []
        classes = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue
                
                cls = parts[0]
                if cls not in ['Car', 'Pedestrian', 'Cyclist']:
                    continue
                
                # Convert KITTI box to [x, y, z, l, w, h, ry]
                box = np.array([float(x) for x in parts[1:8]])
                boxes.append(box)
                classes.append(cls)
        
        if len(boxes) == 0:
            boxes = np.zeros((0, 7), dtype=np.float32)
            classes = np.zeros((0,), dtype=np.int32)
        else:
            boxes = np.stack(boxes, axis=0)
            classes = np.array([self.class_name_to_id(cls) for cls in classes])
        
        return {
            'boxes': boxes,
            'classes': classes
        }
    
    @staticmethod
    def class_name_to_id(name):
        """Convert class name to ID"""
        name_to_id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        return name_to_id.get(name, -1)
