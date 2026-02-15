"""
Dataset class for loading and preprocessing point cloud files with multiple features
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Literal


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
        images_dir: str,
        split: Literal['train', 'val', 'test'] = 'train',
        use_features: list = None,
        num_points: int = None,
        normalize: bool = True,
        use_all_files: bool = False,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_points = num_points
        self.normalize = normalize
        self.use_all_files = use_all_files

        #add the path for the images
        self.images_dir = Path(images_dir)
        
        # Feature selection configuration
        if use_features is None:
            use_features = ['xyz']  # Default to XYZ only
        self.use_features = use_features
        
        # Build channel indices mapping
        self._build_channel_mapping()
        
        # Load all block file paths
        self.block_files = sorted(self.data_dir.glob('**/*.npz'))
        self.image_files = sorted(self.images_dir.glob('**/*.npz'))
        
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
            self.image_files = [self.image_files[i] for i in split_indices]
            
            print(f"Loaded {len(self.block_files)} blocks for {split} split")
    
    def _build_channel_mapping(self):
        """
        Build mapping from feature names to channel indices in the .npz file.
        
        .npz files contain 7 channels: [x, y, z, intensity, return_number, number_of_returns, scan_angle_rank]
        This method creates a list of indices to extract based on use_features.
        """
        # All available channels in preprocessed .npz files
        all_channels = {
            'xyz': [0, 1, 2],         # x, y, z
            'intensity': [3],         # intensity
            'return_number': [4],     # return_number
            'number_of_returns': [5], # number_of_returns
            'scan_angle_rank': [6]    # scan_angle_rank
        }
        
        # Build list of channel indices to extract
        self.channel_indices = []
        for feature in self.use_features:
            if feature not in all_channels:
                raise ValueError(f"Unknown feature: {feature}. Available: {list(all_channels.keys())}")
            self.channel_indices.extend(all_channels[feature])
        
        # Calculate total number of selected channels
        self.num_channels = len(self.channel_indices)
        
        if self.split == 'train':
            print(f"Selected features: {self.use_features}")
            print(f"Channel indices: {self.channel_indices}")
            print(f"Total channels for training: {self.num_channels}")
    
    def _load_image(self, file_path: Path):
        """
        Load a single image from a .npz file 
        Returns
            image: numpy array 256,256 with range z values
        """
        data = np.load(file_path)
        zrange = data['z_range'].astype(np.float32)
        return zrange

    def _load_block(self, file_path: Path):
        """
        Load a single block from .npz file and extract only the configured channels.
        
        Returns:
            points: Array of shape (N, num_selected_channels)
            labels: Array of shape (N,)
        """
        data = np.load(file_path)
        points = data['points'].astype(np.float32)      # Shape: (N, 7)
        labels = data['labels'].astype(np.int64)        # Shape: (N,)
        
        # Select only the configured channels
        points = points[:, self.channel_indices]        # Shape: (N, num_selected_channels)
        
        return points, labels
    
    def _downsample_points(self, points: np.ndarray, labels: np.ndarray):
        """
        Randomly sample num_points from the point cloud.
        If there are fewer points than num_points, repeat sampling with replacement.
        """
        n_points = points.shape[0]
        
        if n_points >= self.num_points:
            # Random sampling without replacement
            indices = np.random.choice(n_points, self.num_points, replace=False)
        else:
            # Random sampling with replacement if we have fewer points
            indices = np.random.choice(n_points, self.num_points, replace=True)
        
        return points[indices], labels[indices]
    
    def _normalize_points(self, points: np.ndarray):
        """
        Normalize XYZ coordinates (first 3 channels) to fit in a unit sphere centered at origin.
        Subtracts centroid and scales by max distance from centroid.
        Additional channels (if present) are left unchanged.
        
        Args:
            points: Array of shape (N, num_channels) where num_channels >= 3
            
        Returns:
            Normalized points with same shape as input
        """
        points = points.copy()
        
        # Normalize only XYZ coordinates (first 3 channels)
        xyz = points[:, :3]
        
        # Center the XYZ points
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        
        # Scale to unit sphere
        max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        if max_dist > 0:
            xyz = xyz / max_dist
        
        # Update XYZ coordinates, keep other channels unchanged
        points[:, :3] = xyz
        
        return points

    def __len__(self):
        """Return the number of blocks in this split"""
        return len(self.block_files)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Returns:
            points: Tensor of shape (num_points, num_channels) or (N, num_channels) if num_points is None
            labels: Tensor of shape (num_points,) or (N,) if num_points is None
        """
        # Load block
        points, labels = self._load_block(self.block_files[idx])
        image = self._load_image(self.image_files[idx])
        # Downsample if needed
        if self.num_points is not None:
            points, labels = self._downsample_points(points, labels)
        
        # Normalize if needed (only affects XYZ coordinates)
        if self.normalize:
            points = self._normalize_points(points)
        
        # Convert to torch tensors
        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).long()
        
        return points, labels, image