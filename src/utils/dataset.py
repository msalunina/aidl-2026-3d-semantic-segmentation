"""
Dataset class for loading and preprocessing LAS point cloud files
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Literal


class DALESDataset(Dataset):
    """
    Dataset for loading DALES point cloud blocks.
    
    Args:
        data_dir: Path to directory containing .npz block files
        split: One of 'train', 'val', or 'test'
        num_points: Number of points to sample per block (if None, use all points)
        normalize: Whether to normalize point clouds to unit sphere
        use_all_files: If True, use all files in directory without splitting (for separate test folder)
        train_ratio: Proportion of data for training (ignored if use_all_files=True)
        val_ratio: Proportion of data for validation (ignored if use_all_files=True)
        test_ratio: Proportion of data for testing (ignored if use_all_files=True)
        seed: Random seed for reproducible splits (ignored if use_all_files=True)
    """
    
    def __init__(
        self, 
        data_dir: str,
        split: Literal['train', 'val', 'test'] = 'train',
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
    
    def _load_block(self, file_path: Path):
        """Load a single block from .npz file"""
        data = np.load(file_path)
        points = data['points'].astype(np.float32)  # Shape: (N, 3)
        labels = data['labels'].astype(np.int64)    # Shape: (N,)
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
        Normalize point cloud to fit in a unit sphere centered at origin.
        Subtracts centroid and scales by max distance from centroid.
        """
        # Center the points
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # Scale to unit sphere
        max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if max_dist > 0:
            points = points / max_dist
        
        return points

    def __len__(self):
        """Return the number of blocks in this split"""
        return len(self.block_files)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Returns:
            points: Tensor of shape (num_points, 3) or (N, 3) if num_points is None
            labels: Tensor of shape (num_points,) or (N,) if num_points is None
        """
        # Load block
        points, labels = self._load_block(self.block_files[idx])
        
        # Downsample if needed
        if self.num_points is not None:
            points, labels = self._downsample_points(points, labels)
        
        # Normalize if needed
        if self.normalize:
            points = self._normalize_points(points)
        
        # Convert to torch tensors
        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).long()
        
        return points, labels