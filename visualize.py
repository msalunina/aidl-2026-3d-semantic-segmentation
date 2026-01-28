# %%
import torch
import torch.nn as nn
from PointNet import ClassificationPointNet
import numpy as np
import random
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from torch.utils.data import random_split
from training_utils import eval_single_epoch


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
    criterion = nn.NLLLoss()

    # FINAL TEST
    # with torch.no_grad():       # Stops tracking gradients, saves memory
    #     test_loss, test_acc = eval_single_epoch(test_loader, network_trained, criterion)
    #     print(f"Final test: loss={test_loss:.4f}, acc={test_acc:.2f}")



    i = 44
    print(test_dataset)
    sample = test_dataset[i]
    points = sample.pos.unsqueeze(0)   #add batch dimension [N, 3] --> [1, N, 3]
  
    with torch.no_grad():       # Stops tracking gradients, saves memory
        log_probs, ix_maxpool, point_features, feature_tnet_tensor, input_tnet_tensor = network_trained(points)
        points_transformed = torch.bmm(points, input_tnet_tensor)                # Batch matrix-matrix product   [batch, nPoints, 3]

    print(input_tnet_tensor[0])
    print("singular values:", torch.linalg.svdvals(input_tnet_tensor[0]))
    print("points_transformed min/max:", points_transformed.min().item(), points_transformed.max().item())


    label = sample.y.item()
    prediction= log_probs.argmax(dim=1).item()
    pts_in = points[0].numpy()
    pts_tn = points_transformed[0].numpy()

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    ax1.scatter(pts_in[:,0], pts_in[:,1], pts_in[:,2], s=2)
    ax1.set_title(f"Input (label={int(label)})")

    ax2.scatter(pts_tn[:,0], pts_tn[:,1], pts_tn[:,2], s=2)
    ax2.set_title(f"After input T-Net (pred={int(prediction)})")

    for ax in (ax1, ax2):
        # ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_box_aspect([1,1,1])  # optional, nicer proportions

    plt.show()


    # Plot 7 samples
    # for i in range(7):

    # fig = plt.figure(figsize=[12,6])
    # # plot input sample
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # sc = ax.scatter(points[:,0], points[:,1], points[:,2], c=points[:,0] ,s=50, marker='o', cmap="viridis", alpha=0.7)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlim3d(-1, 1)
    # ax.title.set_text(f'Input point cloud - Target: {label}')

    # # plot transformation
    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # sc = ax.scatter(points[0,0,:], points[0,1,:], points[0,2,:], c=points[0,0,:] ,s=50, marker='o', cmap="viridis", alpha=0.7)
    # ax.title.set_text(f'Output of "Input Transform" Detected: {preds}')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # plt.savefig(f'figures/Tnet-out-{label}.png',dpi=100)





if __name__ == "__main__":
    main()