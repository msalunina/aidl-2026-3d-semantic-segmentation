"""
Dataset class for loading and preprocessing LAS point cloud files
"""

from pathlib import Path
from torch.utils.data import Dataset

class DALESDataset(Dataset):
    def __init__(self, data_dir: str, normalize=True):
        # read params
        # self.data_dir = Path(data_dir)
        # self.normalize = normalize

        # split into train/val/test sets
        pass

    def _load_data(self):
        pass

    def _downsample(self):
        pass

    def _normalize_points(self):
        pass

    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        # self._load_data()
        # self._downsample()
        # self._normalize_points()
        # return points, addl_features, labels
        pass