import torch
import torch.nn.functional as F
import numpy as np
import sys


class TeeOutput:
    """Write to both stdout and a file"""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        
    def flush(self):
        self.stdout.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()


def sample_point_cloud(point_cloud, num_points):
    """
    Sample a fixed number of points from a point cloud.
    If the point cloud has fewer points than num_points, sample with replacement.
    If it has more points, sample without replacement.
    
    Args:
        point_cloud: Tensor of shape (N, C) where N is number of points, C is number of channels
        num_points: Number of points to sample
    
    Returns:
        Sampled point cloud of shape (num_points, C)
    """
    num_available_points = point_cloud.shape[0]
    
    # Convert to numpy if it's a tensor
    is_tensor = torch.is_tensor(point_cloud)
    if is_tensor:
        device = point_cloud.device
        dtype = point_cloud.dtype
        point_cloud_np = point_cloud.cpu().numpy()
    else:
        point_cloud_np = point_cloud
    
    if num_available_points >= num_points:
        # Sample without replacement
        indices = np.random.choice(num_available_points, num_points, replace=False)
    else:
        # Sample with replacement if not enough points
        indices = np.random.choice(num_available_points, num_points, replace=True)
    
    sampled = point_cloud_np[indices]
    
    # Convert back to tensor if input was a tensor
    if is_tensor:
        sampled = torch.from_numpy(sampled).to(device=device, dtype=dtype)
    
    return sampled


def preprocess_dataset(dataset, num_points):
    """
    Apply point cloud preprocessing to all items in a dataset.
    This includes sampling to a fixed number of points and normalization.
    
    Args:
        dataset: PyTorch Geometric dataset or subset
        num_points: Number of points to sample from each point cloud
    
    Returns:
        Modified dataset with sampled and normalized point clouds
    """
    print(f"Preprocessing point clouds: sampling to {num_points} points and normalizing...")
    
    # Create a wrapper class that applies sampling on-the-fly
    class PreprocessedDataset:
        def __init__(self, base_dataset, num_points):
            self.base_dataset = base_dataset
            self.num_points = num_points
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            data = self.base_dataset[idx]
            # Sample the point cloud
            data.pos = sample_point_cloud(data.pos, self.num_points)
            
            # Normalize point cloud to unit sphere (center at origin, scale to [-1, 1])
            centroid = torch.mean(data.pos, dim=0, keepdim=True)
            data.pos = data.pos - centroid
            max_dist = torch.max(torch.sqrt(torch.sum(data.pos ** 2, dim=1)))
            if max_dist > 0:
                data.pos = data.pos / max_dist
            
            return data
    
    return PreprocessedDataset(dataset, num_points)


def loss_function(predictions, targets, feature_transform, alpha=0.001):
    # Classification loss (NLL loss expects log probabilities from log_softmax)
    # Use ignore_index=-1 to ignore unlabeled points
    classification_loss = F.nll_loss(predictions, targets, ignore_index=-1)
    
    # Regularization loss for feature transform orthogonality
    # feature_transform is [batch_size, 64, 64]
    batch_size = feature_transform.shape[0]
    mat_dim = feature_transform.shape[1]
    
    # Create identity matrix with proper batch dimension
    identity = torch.eye(mat_dim, device=feature_transform.device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Compute A^T * A - I for each batch item and take mean
    mat_diff = torch.bmm(feature_transform, feature_transform.transpose(2, 1)) - identity
    regularization_loss = torch.mean(torch.norm(mat_diff, dim=(1, 2)))
    
    # Total loss
    loss = classification_loss + alpha * regularization_loss
    return loss