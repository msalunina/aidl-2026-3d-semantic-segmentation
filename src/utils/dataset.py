
"""
Dataset class for loading and preprocessing point cloud files with multiple features
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Literal
import os


class DALESDataset(Dataset):
    
    def __init__(
        self, 
        data_dir: str,
        images_dir: str = "",
        use_images: bool = False,
        split: Literal['train', 'val', 'test'] = 'train',
        use_features: list = None,
        num_points: int = None,
        normalize: bool = True,
        augmentation: bool = False,
        rotation_deg_max: float = 180.0,
        use_all_files: bool = False,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.images_dir = Path(images_dir)
        self.split = split
        self.num_points = num_points
        self.normalize = normalize
        self.use_all_files = use_all_files
        self.augmentation = augmentation and split == 'train'
        self.rotation_deg_max = rotation_deg_max
        self.use_images = use_images

        if use_features is None:
            use_features = ['xyz']
        self.use_features = use_features
        
        self._build_channel_mapping()
        
        self.block_files = sorted(self.data_dir.glob('**/*.npz'))
        
        if len(self.block_files) == 0:
            raise ValueError(f"No .npz files found in {self.data_dir}")
        
        if use_all_files:
            print(f"Loaded {len(self.block_files)} blocks from {self.data_dir}")
        else:
            np.random.seed(seed)
            indices = np.random.permutation(len(self.block_files))
            
            train_end = int(len(indices) * train_ratio)
            val_end = train_end + int(len(indices) * val_ratio)
            
            if split == 'train':
                split_indices = indices[:train_end]
            elif split == 'val':
                split_indices = indices[train_end:val_end]
            elif split == 'test':
                split_indices = indices[val_end:]
            else:
                raise ValueError(f"Invalid split: {split}")
            
            self.block_files = [self.block_files[i] for i in split_indices]
            print(f"Loaded {len(self.block_files)} blocks for {split} split")
    
    def _build_channel_mapping(self):
        
        all_channels = {
            'xyz': [0, 1, 2],
            'intensity': [3],
            'return_number': [4],
            'number_of_returns': [5],
            'scan_angle_rank': [6]
        }
        
        self.channel_indices = []
        for feature in self.use_features:
            if feature not in all_channels:
                raise ValueError(f"Unknown feature: {feature}")
            self.channel_indices.extend(all_channels[feature])
        
        self.num_channels = len(self.channel_indices)
    
    def _load_image(self, file_path):
        
        data = np.load(file_path)
        
        #c1 = data['density'].astype(np.float32)
        #c2 = data['z_max'].astype(np.float32)
        #c3 = data['z_mean'].astype(np.float32)
        #c4 = data['z_range'].astype(np.float32)
        
        c1 = data['density'].astype(np.float32)
        c1 = (c1 - c1.min()) / (c1.max() - c1.min())
        c2 = data['z_max'].astype(np.float32)
        c2 = (c2 - c2.min()) / (c2.max() - c2.min())
        c3 = data['z_mean'].astype(np.float32)
        c3 = (c3 - c3.min()) / (c3.max() - c3.min())
        c4 = data['z_range'].astype(np.float32)
        c4 = (c4 - c4.min()) / (c4.max() - c4.min())

        return np.stack([c1, c2, c3, c4], axis=-1) 

    def _load_block(self, file_path: Path):
        
        data = np.load(file_path)

        points = data['points'].astype(np.float32)
        labels = data['labels'].astype(np.int64)
        points = points[:, self.channel_indices]

        image_name = ""
        if self.use_images:
            image_name = data['bev_filename'].item()

        xy_grid = None
        if 'xy_grid' in data.files:
            xy_grid = data['xy_grid'].astype(np.float32)

        return points, labels, image_name, xy_grid
    
    def _downsample_points(self, points, labels, xy_grid=None):
        
        n_points = points.shape[0]
        
        if n_points >= self.num_points:
            indices = np.random.choice(n_points, self.num_points, replace=False)
        else:
            indices = np.random.choice(n_points, self.num_points, replace=True)
        
        points = points[indices]
        labels = labels[indices]

        if xy_grid is not None:
            xy_grid = xy_grid[indices]
            return points, labels, xy_grid

        return points, labels
    
    def _normalize_points(self, points):
        
        points = points.copy()
        xyz = points[:, :3]
        
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        
        max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        if max_dist > 0:
            xyz = xyz / max_dist
        
        points[:, :3] = xyz
        return points

    def _augment_points(self, points):
        
        points = points.copy()
        xyz = points[:, :3]

        angle_rad = np.deg2rad(np.random.uniform(-180, 180))
        cos_theta, sin_theta = np.cos(angle_rad), np.sin(angle_rad)

        rotation = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        xyz = np.matmul(xyz, rotation.T)
        points[:, :3] = xyz

        return points

    def __len__(self):
        return len(self.block_files)
    
    def __getitem__(self, idx):
        
        points, labels, image_name, xy_grid = self._load_block(self.block_files[idx])

        if self.use_images:
            image = self._load_image(Path(os.path.join(self.images_dir, image_name)))
            if xy_grid is None:
                raise ValueError("xy_grid not found in block file")

        if self.num_points is not None:
            if self.use_images:
                points, labels, xy_grid = self._downsample_points(points, labels, xy_grid)
            else:
                points, labels = self._downsample_points(points, labels)
        
        if self.normalize:
            points = self._normalize_points(points)

        if self.augmentation:
            points = self._augment_points(points)
        
        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).long()

        if self.use_images:
            xy_grid = torch.from_numpy(xy_grid).float()
            return points, labels, image, xy_grid
        else:
            return points, labels
