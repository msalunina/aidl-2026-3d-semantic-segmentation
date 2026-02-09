import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
from utils_training import unpack_batch, set_seed
from utils_data import load_dataset, choose_architecture

def main():

    # FORCE CPU
    device = torch.device("cpu")
    
    # LOAD CHECKPOINT
    # RUN_NAME = "ClassPointNet_ShapeNet_1024pts_1epochs"
    RUN_NAME = "ClassPointNet_ModelNet_1024pts_10epochs"
    # RUN_NAME = "ClassPointNetSmall_ShapeNet_1024pts_1epochs"
    # RUN_NAME = "ClassPointNetSmall_ModelNet_1024pts_30epochs"
    run_dir = Path("runs") / RUN_NAME
    checkpoint_path = run_dir / "checkpoint.pt"

    checkpoint_state = torch.load(checkpoint_path, map_location=device, weights_only=False)    # it loads more things that weights
    config = checkpoint_state["config"]  

    print(config)
    # SEEDS i histories varies
    set_seed(config["seed"])

    # DATASET 
    _, _, test_dataset, id_to_name = load_dataset(config)
    num_classes = len(id_to_name)

    # CHOOSE AND UPDATE ARCHITECTURE
    network_trained = choose_architecture(config["architecture"], num_classes)
    network_trained.load_state_dict(checkpoint_state["model"])
    network_trained.eval()      # changes behaviour of some layers (e.g. dropout off), does not strop gradient

    # GET A SAMPLE, SHAPE IT AND MAKE IT THROUGH THE NETWORK
    print(f"Number of test samples: {len(test_dataset)}")

    sample = 200
    points, label = unpack_batch(test_dataset[sample])
    if points.dim() == 2:
        points = points.unsqueeze(0)
  
    with torch.no_grad():               # Stops tracking gradients, saves memory
        log_probs, ix_maxpool, _, _, input_tnet = network_trained(points)
        points_transformed = torch.bmm(points, input_tnet)                # Batch matrix-matrix product   [batch, nPoints, 3]

    print(input_tnet[0])
    print("singular values:", torch.linalg.svdvals(input_tnet[0]))

    # ------------------------------------------------
    # PLOTING POINTS BEFORE AND AFTER INPUT TRANSFORM
    # ------------------------------------------------
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
    ax1.set_title(f"Input (label={int(label)}={id_to_name[int(label)]})")
    ax2.scatter(pts_tn[:,0], pts_tn[:,1], pts_tn[:,2], s=2)
    ax2.set_title(f"After input T-Net (pred={int(prediction)}={id_to_name[int(prediction)]})")
    ax3.scatter(critical_points[:,0], critical_points[:,1], critical_points[:,2], s=2, color="red")
    ax3.set_title(f"Critical points (maximize features)")
    for ax in (ax1, ax2, ax3):
        ax.set_aspect('equal')
        # ax.set_box_aspect([1,1,1])  

    plt.show()


if __name__ == "__main__":
    main()