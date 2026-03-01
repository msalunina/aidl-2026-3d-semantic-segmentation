"""
Class-balanced sampler for oversampling blocks containing rare classes.
"""

import numpy as np
import torch
from torch.utils.data import Sampler
from tqdm import tqdm


class ClassBalancedSampler(Sampler):
    """
    Sampler that oversamples blocks containing rare classes.
    
    This sampler computes a sampling weight for each block based on whether
    it contains rare classes (e.g., Vehicle, Utility). Blocks with rare classes
    are more likely to be sampled during training.
    
    Args:
        dataset: DALESDataset instance
        rare_classes: List of class indices to oversample (default: [3, 4] for Vehicle and Utility)
        rare_class_boost: Multiplier for sampling weight of blocks with rare classes (default: 3.0)
        verbose: Print statistics about class distribution (default: True)
    """
    
    def __init__(self, dataset, rare_classes=None, rare_class_boost=3.0, verbose=True):
        self.dataset = dataset
        self.rare_class_boost = rare_class_boost
        self.verbose = verbose
        
        # Default rare classes: Vehicle (3) and Utility (4)
        if rare_classes is None:
            rare_classes = [3, 4]
        self.rare_classes = rare_classes
        
        # Calculate sampling weights
        self.weights = self._calculate_weights()
        
    def _calculate_weights(self):
        """Calculate sampling weight for each block based on class distribution."""
        weights = []
        blocks_with_rare = 0
        
        if self.verbose:
            print(f"\nAnalyzing {len(self.dataset.block_files)} blocks for class-balanced sampling...")
            iterator = tqdm(self.dataset.block_files, desc="Computing sampling weights")
        else:
            iterator = self.dataset.block_files
        
        for block_file in iterator:
            data = np.load(block_file)
            labels = data['labels']
            
            # Check if block contains any rare classes
            has_rare_class = False
            for rare_class in self.rare_classes:
                if np.any(labels == rare_class):
                    has_rare_class = True
                    break
            
            # Assign weight based on presence of rare classes
            if has_rare_class:
                weight = self.rare_class_boost
                blocks_with_rare += 1
            else:
                weight = 1.0
            
            weights.append(weight)
        
        if self.verbose:
            pct_with_rare = 100.0 * blocks_with_rare / len(weights)
            print(f"Blocks with rare classes: {blocks_with_rare}/{len(weights)} ({pct_with_rare:.1f}%)")
            print(f"Rare class boost factor: {self.rare_class_boost}x")
            print(f"Effective oversampling ratio: {blocks_with_rare * self.rare_class_boost / len(weights):.2f}x")
        
        return torch.FloatTensor(weights)
    
    def __iter__(self):
        """
        Sample indices according to weights.
        
        Uses multinomial sampling with replacement, so blocks with rare classes
        will be sampled multiple times per epoch.
        """
        # Sample with replacement according to weights
        indices = torch.multinomial(
            self.weights, 
            num_samples=len(self.dataset),
            replacement=True
        )
        return iter(indices.tolist())
    
    def __len__(self):
        return len(self.dataset)
