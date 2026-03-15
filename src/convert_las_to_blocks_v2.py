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
    """
    Dataset for loading DALES point cloud blocks with configurable feature selection.
    
    The dataset loads .npz files containing all available features (7 channels:
    x, y, z, intensity, return_number, number_of_returns, scan_angle_rank) and selects only the
    requested features for training based on configuration.
    
    Channel mapping in .npz files:
    - 0-2: x, y, z coordinates
    - 3: intensity
    - 4: return_number
    - 5: number_of_returns
    - 6: scan_angle_rank
    
    Args:
        data_dir: Path to directory containing .npz block files
        split: One of 'train', 'val', or 'test'
        use_features: List of feature names to use ['xyz', 'intensity', 'return_number', 'number_of_returns', 'scan_angle_rank']
        num_points: Number of points to sample per block (if None, use all points)
        normalize: Whether to normalize XYZ coordinates to unit sphere (other channels unchanged)
        use_all_files: If True, use all files in directory without splitting (for separate test folder)
        train_ratio: Proportion of data for training (ignored if use_all_files=True)
        val_ratio: Proportion of data for validation (ignored if use_all_files=True)
        test_ratio: Proportion of data for testing (ignored if use_all_files=True)
        seed: Random seed for reproducible splits (ignored if use_all_files=True)
    """
    
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
        scale_min: float = 0.9,
        scale_max: float = 1.1,
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
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.use_images = use_images

        if self.scale_min <= 0:
            raise ValueError(f"scale_min must be > 0, got {self.scale_min}")
        if self.scale_max < self.scale_min:
            raise ValueError(f"scale_max ({self.scale_max}) must be >= scale_min ({self.scale_min})")
        
        # Feature selection configuration
        if use_features is None:
            use_features = ['xyz']  # Default to XYZ only
        self.use_features = use_features
        
        # Build channel indices mapping
        self._build_channel_mapping()
        
        # Load all block file paths
        self.block_files = sorted(self.data_dir.glob('**/*.npz'))
        
        if len(self.block_files) == 0:
            raise ValueError(f"No .npz files found in {self.data_dir}")
        
        # If use_all_files is True, skip splitting (useful for separate test folder)
        if use_all_files:
            print(f"Loaded {len(self.block_files)} blocks from {self.data_dir}")
        else:
            if not np.isclose(train_ratio + val_ratio + (1 - train_ratio - val_ratio), 1.0):
                raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")
            
            # Split data into train/val/test
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
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
            
            self.block_files = [self.block_files[i] for i in split_indices]
            
            print(f"Loaded {len(self.block_files)} blocks for {split} split")
    
    def _build_channel_mapping(self):
        """
        Build mapping from feature names to channel indices in the .npz file.
        
        .npz files contain 7 channels: [x, y, z, intensity, return_number, number_of_returns, scan_angle_rank]
        This method creates a list of indices to extract based on use_features.
        """
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
                raise ValueError(f"Unknown feature: {feature}. Available: {list(all_channels.keys())}")
            self.channel_indices.extend(all_channels[feature])
        
        self.num_channels = len(self.channel_indices)
        
        if self.split == 'train':
            print(f"Selected features: {self.use_features}")
            print(f"Channel indices: {self.channel_indices}")
            print(f"Total channels for training: {self.num_channels}")
            print(
                f"Augmentation: {self.augmentation} "
                f"(rotation_deg_max={self.rotation_deg_max}, scale=[{self.scale_min}, {self.scale_max}])"
            )
    
    def _load_image(self, file_path):
        """
        Load a single image from a .npz file 
        
        Returns:
            image: numpy array [256, 256, 4]
        """
        data = np.load(file_path)
        c1 = data['density'].astype(np.float32)
        c1 = (c1 - c1.min()) / (c1.max() - c1.min() + 1e-8)

        c2 = data['z_max'].astype(np.float32)
        c2 = (c2 - c2.min()) / (c2.max() - c2.min() + 1e-8)

        c3 = data['z_mean'].astype(np.float32)
        c3 = (c3 - c3.min()) / (c3.max() - c3.min() + 1e-8)

        c4 = data['z_range'].astype(np.float32)
        c4 = (c4 - c4.min()) / (c4.max() - c4.min() + 1e-8)

        return np.stack([c1, c2, c3, c4], axis=-1)

    def _load_block(self, file_path: Path):
        """
        Load a single block from .npz file and extract only the configured channels.
        
        Returns:
            points: Array of shape (N, num_selected_channels)
            labels: Array of shape (N,)
            image_name: BEV filename
            xy_grid: Array of shape (N, 2) with coordinates in [-1, 1] for grid_sample,
                     or None if not present
        """
        data = np.load(file_path)

        points = data['points'].astype(np.float32)      # Shape: (N, 7)
        labels = data['labels'].astype(np.int64)        # Shape: (N,)
        points = points[:, self.channel_indices]        # Shape: (N, num_selected_channels)

        image_name = data['bev_filename'].item()

        xy_grid = None
        if 'xy_grid' in data.files:
            xy_grid = data['xy_grid'].astype(np.float32)

        return points, labels, image_name, xy_grid
    
    def _downsample_points(self, points: np.ndarray, labels: np.ndarray, xy_grid: np.ndarray = None):
        """
        Randomly sample num_points from the point cloud.
        If there are fewer points than num_points, repeat sampling with replacement.
        """
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
    
    def _normalize_points(self, points: np.ndarray):
        """
        Normalize XYZ coordinates (first 3 channels) to fit in a unit sphere centered at origin.
        Additional channels (if present) are left unchanged.
        """
        points = points.copy()
        
        xyz = points[:, :3]
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        
        max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        if max_dist > 0:
            xyz = xyz / max_dist
        
        points[:, :3] = xyz
        return points

    def _augment_points(self, points: np.ndarray):
        """
        Apply random geometric augmentation to XYZ coordinates.

        Augmentations:
        - Random rotation around Z-axis
        - Random isotropic scaling
        """
        points = points.copy()

        if points.shape[1] < 3:
            return points

        xyz = points[:, :3]

        angle_rad = np.deg2rad(np.random.uniform(-self.rotation_deg_max, self.rotation_deg_max))
        cos_theta, sin_theta = np.cos(angle_rad), np.sin(angle_rad)
        rotation = np.array([
            [cos_theta, -sin_theta, 0.0],
            [sin_theta,  cos_theta, 0.0],
            [0.0,        0.0,       1.0],
        ], dtype=np.float32)

        xyz = np.matmul(xyz, rotation.T)
        points[:, :3] = xyz
        return points

    def __len__(self):
        """Return the number of blocks in this split"""
        return len(self.block_files)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Returns:
            If use_images=False:
                points, labels
            If use_images=True:
                points, labels, image, xy_grid
        """
        points, labels, image_name, xy_grid = self._load_block(self.block_files[idx])

        if self.use_images:
            image = self._load_image(Path(os.path.join(self.images_dir, image_name)))
            if xy_grid is None:
                raise ValueError(
                    f"Block file {self.block_files[idx]} does not contain 'xy_grid'. "
                    f"Please regenerate block files with the updated preprocessing script."
                )

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