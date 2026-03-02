import argparse
import torch
import numpy as np
import os
import random
from utils.config_parser import ConfigParser
from utils.dataset import DALESDataset
from utils.evaluator import test_model_segmentation
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

    # Set device
    device = set_device()
    
    base_path = Path(os.getcwd())
    if "src" in base_path.parts:
        base_path = base_path[:-1]

    epoch = 0    
    checkpoint_path = base_path / "snapshots" / "PointNet" / f"pointnet_{epoch}_epochs.pt"
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)    # it loads more things that weights
    config = checkpoint["config"]
    COLOR_MAP = config.viz_2d['color_mapping']

    # set seed (here? config.dataset_seed?)
    set_seed(config.dataset_seed)

    print("\n" + "="*60)
    print("CREATING DATASETS AND DATALOADERS")
    print("="*60)
    # Create test datasets
    test_dataset = DALESDataset(
        data_dir=f"{config.model_data_path}/test",
        images_dir=f"{config.image_data_path}/test",
        split='test',
        use_features=config.dataset_use_features,
        num_points=config.train_num_points,
        normalize=config.dataset_normalize,
        use_all_files=config.dataset_test_use_all_files,
        seed=config.dataset_seed
    )

    print("\n" + "="*60)
    print("INITIALIZING MODEL, LOSS and OPTIMIZER")
    print("="*60)
    # Initialize model dependent on the provided input
    if config.model_name == "pointnet":
        from models.pointnet import PointNetSegmentation
        model_trained = PointNetSegmentation(num_classes=config.num_classes, 
                                             input_channels=config.num_channels, 
                                             dropout=config.dropout_rate).to(device)
    elif config.model_name == "ipointnet":
        from models.pointnet import IPointNetSegmentation
        model_trained = IPointNetSegmentation(num_classes=config.num_classes, 
                                              input_channels=config.num_channels, 
                                              dropout=config.dropout_rate).to(device)
    else: 
        raise ValueError(f"Model name {config.model_name} does not exist")

    # UPDATE ARCHITECTURE
    model_trained.load_state_dict(checkpoint["model_state_dict"])
    model_trained.to(device)
    model_trained.eval()      # changes behaviour of some layers (e.g. dropout off, batchnorm), does not strop gradient

    print("\n" + "="*60)
    print("FORWARDING BLOCK")
    print("="*60)
    # Choosing block and making it through the network
    block = 44  
    points, labels, _ = test_dataset[block]
    points_BNC = points.unsqueeze(0)            # add batch dimension
    points_BNC = points_BNC.to(device)          # sent to same device as model
    labels = labels.cpu().numpy().astype(int)   # no need to be moved to device, but convert to numpy

    
    print("points shape from dataset:", points_BNC.shape)   # expect [N, 6]

    with torch.no_grad():                       # Stops tracking gradients, saves memory
        feature_tnet, log_probs_BCN = model_trained(points_BNC)   

    preds_BN = log_probs_BCN.argmax(dim=1)                  # [B, N]
    preds = preds_BN.squeeze(0).cpu().numpy().astype(int)   # [N]
    pts_in = points_BNC[0,:,:].cpu().numpy()                # [N, C]      


    id_valid = (labels != config.ignore_label)  
    id_correct = id_valid & (preds == labels)
    id_wrong = id_valid & (preds != labels)

    num_valid = id_valid.sum()
    num_correct = id_correct.sum()
    num_wrong = id_wrong.sum()
    print("Valid points:", num_valid, "\nCorrect:", num_correct, "\nWrong:", num_wrong)


    # ------------------------------------------------
    #                   PLOTING 
    # ------------------------------------------------
    fig1 = plt.figure(figsize=(12, 6))

    # PLOTING POINTS BEFORE AND AFTER INPUT TRANSFORM
    ax1 = fig1.add_subplot(121, projection="3d")
    for cls in sorted(set(labels)):
        m = (labels == cls)
        ax1.scatter(pts_in[m, 0], pts_in[m, 1], pts_in[m, 2], s=2, c=COLOR_MAP.get(cls, "black"),label=str(cls))
    ax1.set_title(f"Input points")

    # PLOTING CORRECT vs WRONG POINTS
    ax2 = fig1.add_subplot(122, projection="3d")
    ax2.scatter(pts_in[id_correct, 0], pts_in[id_correct, 1], pts_in[id_correct, 2], s=2, c="black", label="correct")
    ax2.scatter(pts_in[id_wrong, 0], pts_in[id_wrong, 1], pts_in[id_wrong, 2], s=2, c="red", label="wrong")
    ax2.set_title(f"Correct (black) / Wrong (red): {num_correct}/{num_wrong} predictions")
    ax2.legend()
    
    for ax in (ax1, ax2):
        ax.set_aspect('equal')
    plt.show()
