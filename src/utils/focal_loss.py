"""
Focal Loss for addressing class imbalance in semantic segmentation.
Based on: Focal Loss for Dense Object Detection (Lin et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    The focal loss focuses training on hard examples while down-weighting
    easy examples. This is particularly effective for class imbalance.
    
    Args:
        alpha: Class weights (tensor of shape [num_classes]). If None, no weighting.
        gamma: Focusing parameter (default: 2.0). Higher gamma puts more focus on hard examples.
        ignore_index: Label value to ignore in loss computation (default: -1).
    """
    
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights (shape: [num_classes])
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, log_probs, targets):
        """
        Compute focal loss.
        
        Args:
            log_probs: [B, C, N] - log probabilities from model (output of log_softmax)
            targets: [B, N] - ground truth labels
            
        Returns:
            Scalar loss value
        """
        # Convert log_probs to probabilities
        probs = torch.exp(log_probs)  # [B, C, N]
        
        B, C, N = probs.shape
        
        # Reshape for easier processing
        probs = probs.permute(0, 2, 1).contiguous()  # [B, N, C]
        probs = probs.view(-1, C)  # [B*N, C]
        targets_flat = targets.view(-1)  # [B*N]
        
        # Mask out ignored indices
        valid_mask = (targets_flat != self.ignore_index)
        valid_targets = targets_flat[valid_mask]
        valid_probs = probs[valid_mask]
        
        if valid_targets.numel() == 0:
            # No valid targets, return zero loss
            return torch.tensor(0.0, device=probs.device, requires_grad=True)
        
        # Get probability of correct class for each point
        # Create one-hot encoding
        targets_one_hot = F.one_hot(valid_targets, num_classes=C).float()  # [N_valid, C]
        
        # Get probability of ground truth class
        pt = (valid_probs * targets_one_hot).sum(dim=1)  # [N_valid]
        pt = pt.clamp(min=1e-8, max=1.0)  # Numerical stability
        
        # Compute focal term: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Compute cross entropy: -log(pt)
        ce = -torch.log(pt)
        
        # Apply class weights if provided
        if self.alpha is not None:
            # Get alpha for each sample based on its ground truth class
            alpha_t = self.alpha[valid_targets]  # [N_valid]
            ce = alpha_t * ce
        
        # Apply focal weight
        loss = focal_weight * ce
        
        return loss.mean()
