import torch
import torch.nn as nn
from PointNet import ClassificationPointNet
import numpy as np
import random
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
import os


 # SEEDS i histories varies
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# FORCE CPU
device = torch.device("cpu")

def main():
    config = {"epochs": 10,
                "lr": 0.001,
                "batch_size": 32,
                "nPoints": 1024} 

    # GET NAME CLASSES
    path_folder = "data/ModelNet/raw"
    name_classes = sorted( d for d in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, d)))

    # TRANSFORMS
    transform = T.Compose([
        T.SamplePoints(config["nPoints"]),      # mesh -> point cloud (pos: [N,3])
        T.NormalizeScale(),                     # center + scale to unit sphere
    ])

    # DATASET + DATALOADER
    # "pre_transform" is processed only once and saved to processed/*.pt forever (until deleted)
    # "transform" is processed every time __getitem__ is called, every time an object is called (i.e. it samples it)
    test_dataset  = ModelNet(root="data/ModelNet", name="10", train=False, pre_transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False) 

    # LOAD TRAINED NETWORK
    save_path = "checkpoints/ClassificationPointnet_10epochs_1024.pt"
    checkpoint_state = torch.load(save_path, map_location=device)
    network_trained = ClassificationPointNet(num_classes=10)
    network_trained.load_state_dict(checkpoint_state["model"])
    network_trained.eval()      # changes behaviour of some layers (e.g. dropout off), does not strop gradient

    # GET A SAMPLE, SHAPE IT AND MAKE IT THROUGH THE NETWORK
    print(f"Number of test samples: {len(test_dataset)}")
    i = 10
    sample = test_dataset[i]            # [N, 3]
    points = sample.pos.unsqueeze(0)    # Network needs a batch dimension:; [N, 3] --> [1, N, 3]
  
    with torch.no_grad():       # Stops tracking gradients, saves memory
        log_probs, ix_maxpool, _, _, input_tnet = network_trained(points)
        points_transformed = torch.bmm(points, input_tnet)                # Batch matrix-matrix product   [batch, nPoints, 3]

    print(input_tnet[0])
    print("singular values:", torch.linalg.svdvals(input_tnet[0]))
    print("points_transformed min/max:", points_transformed.min().item(), points_transformed.max().item())

    # ------------------------------------------------
    # PLOTING POINTS BEFORE AND AFTER INPUT TRANSFORM
    # ------------------------------------------------
    label = sample.y.item()
    prediction= log_probs.argmax(dim=1).item()
    pts_in = points[0,:,:].numpy()
    pts_tn = points_transformed[0,:,:].numpy()
    # ------------------------------------------------
    #        PLOTING REPRESENTATIVE POINTS 
    #      (contributing to global features)
    # ------------------------------------------------
    ix = ix_maxpool[0, :, 0]
    list_indices = torch.unique(ix).numpy()
    critical_points = pts_in[list_indices,:]

    fig1 = plt.figure(figsize=(12, 6))
    ax1 = fig1.add_subplot(131, projection="3d")
    ax2 = fig1.add_subplot(132, projection="3d")
    ax3 = fig1.add_subplot(133, projection="3d")
    ax1.scatter(pts_in[:,0], pts_in[:,1], pts_in[:,2], s=2)
    ax1.set_title(f"Input (label={int(label)}={name_classes[label]})")
    ax2.scatter(pts_tn[:,0], pts_tn[:,1], pts_tn[:,2], s=2)
    ax2.set_title(f"After input T-Net (pred={int(prediction)}={name_classes[prediction]})")
    ax3.scatter(critical_points[:,0], critical_points[:,1], critical_points[:,2], s=2, color="red")
    ax3.set_title(f"Critical points (maximize features)")
    for ax in (ax1, ax2, ax3):
        ax.set_box_aspect([1,1,1])  

  
    plt.show()


if __name__ == "__main__":
    main()