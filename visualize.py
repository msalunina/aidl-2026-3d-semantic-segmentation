import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
import os
from training_utils import unpack_batch



def load_dataset(config):
    
    if config["dataset"] == "ModelNet":    
        import torch_geometric.transforms as T
        from torch_geometric.datasets import ModelNet
        from torch_geometric.loader import DataLoader

        data_path = "data/ModelNet"

        transform = T.Compose([T.SamplePoints(config["nPoints"]),      # mesh -> point cloud (pos: [N,3])
                               T.NormalizeScale()])                    # center + scale to unit sphere
        # "pre_transform" is processed only once and saved to processed/*.pt forever (until deleted)
        # "transform" is processed every time __getitem__ is called, every time an object is called (i.e. it samples it)
        dataset = ModelNet(root=data_path, name="10", train=False, pre_transform=transform)    
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)    

        # # GET NAME CLASSES
        folder_path = os.path.join(data_path, "raw")
        name_classes = sorted( d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d)))


    elif config["dataset"] == "ShapeNet":
        from utils.shapenet_dataset_ebp import shapeNetDataset
        from torch.utils.data import DataLoader   # <-- instead of torch_geometric.loader

        data_path = "data/ShapeNet/PartAnnotation"
        dataset = shapeNetDataset(dataset_path=data_path, point_cloud_size=config["nPoints"], mode=2, class_name="")
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)    

        id_to_class = {v: k for k, v in dataset._object_classes.items()}       # CREATES A DICTIONARY
        name_classes = [id_to_class[i] for i in range(len(id_to_class))]       # CREATES A LIST        
    
    else:
        raise TypeError(f"No idea what is dataset {config['dataset']}")
    
    return dataset, loader, name_classes


def choose_architecture(config):
    if config["architecture"] == "PointNet":    
        from PointNet import ClassificationPointNet
    elif config["architecture"] == "PointNetSmall": 
        from PointNetSmall import ClassificationPointNet
    else:
        raise TypeError(f"No idea what is architecture {config['architecture']}")

    network = ClassificationPointNet(num_classes=config["nClasses"])

    return network



def main():

    # SEEDS i histories varies
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # FORCE CPU
    device = torch.device("cpu")

    # FUTURE: config = checkpoint_state["config"]       
    config = {"dataset": "ShapeNet",
              "nClasses": 16,
              "nPoints": 1024,
              "architecture": "PointNetSmall",
              "epochs": 10,
              "lr": 0.001,
              "batch_size": 32} 
    
    # LOAD CHECKPOINT
    # save_path = "checkpoints/ClassificationPointnet_ModelNet_10epochs_1024.pt"
    save_path = "checkpoints/ClassificationPointnetSmall_ShapeNet_10epochs_1024.pt"
    checkpoint_state = torch.load(save_path, map_location=device)

    # LOADING DATASET 
    test_dataset, _ , name_classes = load_dataset(config)

    # CHOOSE AND UPDATE ARCHITECTURE
    network_trained = choose_architecture(config)
    network_trained.load_state_dict(checkpoint_state["model"])
    network_trained.eval()      # changes behaviour of some layers (e.g. dropout off), does not strop gradient

    # GET A SAMPLE, SHAPE IT AND MAKE IT THROUGH THE NETWORK
    print(f"Number of test samples: {len(test_dataset)}")

    sample = 800
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