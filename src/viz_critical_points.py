import argparse
import torch
import numpy as np
import os
import random
from utils.config_parser import ConfigParser
from utils.dataset import DALESDataset
from torch.utils.data import DataLoader
from utils.trainer import train_model_segmentation
from pathlib import Path
import matplotlib.pyplot as plt


def set_device():
    if torch.cuda.is_available(): 
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): 
        device = torch.device("mps")
    else: device = torch.device("cpu")
    
    print(f"\nUsing device: {device}")
    return device


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    
    device = set_device()
    
    config_parser = ConfigParser(
        default_config_path="config/default.yaml",
        parser=argparse.ArgumentParser(description='3D Semantic Segmentation on DALES Dataset')
    )
    config = config_parser.load()
    COLOR_MAP = config.viz_2d['color_mapping']

    set_seed(config.dataset_seed)

    print("\n" + "="*60)
    print("CREATING DATASETS AND DATALOADERS")
    print("="*60)
    # Create datasets
    train_dataset = DALESDataset(
        data_dir=f"{config.model_data_path}/train",
        images_dir=f"{config.image_data_path}/train",
        split='train',
        use_features=config.dataset_use_features,
        num_points=config.train_num_points,
        normalize=config.dataset_normalize,
        train_ratio=config.dataset_train_ratio,
        val_ratio=config.dataset_val_ratio,
        seed=config.dataset_seed
    )

    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    # Initialize model dependent on the provided input
    from models.pointnet_analysis import PointNetSegmentation
    model = PointNetSegmentation(num_classes=config.num_classes, input_channels=config.num_channels, dropout=config.dropout_rate).to(device)
    model.eval()                                # changes behaviour of some layers (e.g. dropout off, batchnorm)

    print("\n" + "="*60)
    print("FORWARDING BLOCK")
    print("="*60)
    # Choosing block and making it through the network
    block = 44  
    points, labels, _ = train_dataset[block]
    points_BNC = points.unsqueeze(0)            # add batch dimension
    points_BNC = points_BNC.to(device)          # sent to same device as model
    labels = labels.cpu().numpy().astype(int)   # no need to be moved to device, but convert to numpy

    with torch.no_grad():                       # Stops tracking gradients, saves memory
        feature_tnet, log_probs_BCN, critical_point_indices, input_transform = model(points_BNC)            

    
    # Points before and after iput_transform
    points_transformed = torch.bmm(points_BNC[:, :, :3] , input_transform)                # Batch matrix-matrix product   [batch, nPoints, 3]
    pts_in = points_BNC[0,:,:].cpu().numpy()    
    pts_tn = points_transformed[0,:,:].cpu().numpy() 
    # Critical points and its labels
    ix = critical_point_indices[0, :, 0]
    list_indices = torch.unique(ix).cpu().numpy()
    critical_points = pts_in[list_indices,:]   
    critical_labels = labels[list_indices]

    # ------------------------------------------------
    #                   PLOTING 
    # ------------------------------------------------
    fig1 = plt.figure(figsize=(12, 6))

    # PLOTING POINTS BEFORE AND AFTER INPUT TRANSFORM
    ax1 = fig1.add_subplot(131, projection="3d")
    ax2 = fig1.add_subplot(132, projection="3d")
    for cls in sorted(set(labels)):
        m = labels == cls
        ax1.scatter(pts_in[m, 0], pts_in[m, 1], pts_in[m, 2], s=2, c=COLOR_MAP.get(cls, "black"),label=str(cls))
        ax2.scatter(pts_tn[m, 0], pts_tn[m, 1], pts_tn[m, 2], s=2, c=COLOR_MAP.get(cls, "black"),label=str(cls))
    ax1.set_title(f"Input points")
    ax2.set_title(f"Transformed points (after input T-Net)")

    # PLOTING REPRESENTATIVE POINTS (contributing to global features)
    ax3 = fig1.add_subplot(133, projection="3d")
    for cls in sorted(set(labels)):
        m = critical_labels == cls
        ax3.scatter(critical_points[m, 0], critical_points[m, 1], critical_points[m, 2], s=2, c=COLOR_MAP.get(cls, "black"),label=str(cls))
    ax3.set_title(f"Critical points (maximize features)")
    

    for ax in (ax1, ax2, ax3):
        ax.set_aspect('equal')
        # ax.set_box_aspect([1,1,1])  
    plt.show()
