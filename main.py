# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from torch.utils.data import random_split
import random
import os
from PointNet import ClassificationPointNet, BasePointNet, TransformationNet
from training_utils import train_model
import time


def info_dataset_batch(dataset, data_loader, name_classes):
    # There are 32 objects per batch, and each object has 1024 points. 
    # But it does not separate objects....all 32x1024 points are stacked together.
    # batch is a Data object (like a struct in matlab), with attributes. The atributes are:
    # batch.pos   --> llista de punts per batch ([32x1024,3])
    # batch.batch --> integer telling each point to which object it belongs to.
    # batch.y     --> labels, one label per object (not point!!!)
    # batch.ptr   --> ni puta idea
    # They can also be accessed like a dictionary: batch["pos"], batch["y"]...

    # info item from a batch
    batch = next(iter(data_loader))
    print(f"BATCH: {batch}")
    batchItem = 30
    pointsObjectID = batch.pos[batch.batch==batchItem]
    print("Item batch shape:", pointsObjectID.shape, "Label:", batch.y[batchItem])

    # info item from the original dataset
    datasetItem = 44
    pos = dataset[datasetItem].pos.cpu().numpy()
    label = dataset[datasetItem].y.item()

    fig = plt.figure(figsize=[7,7])
    ax = plt.axes(projection='3d')
    sc = ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=pos[:,0] ,s=80, marker='o', cmap="viridis", alpha=0.7)
    ax.set_zlim3d(-1, 1)
    plt.title(f'Label: {label} = {name_classes[label]}')
    plt.show()

# RUN ONLY IF EXECUTED AS MAIN
if __name__ == "__main__":

    save_path = "checkpoints/ClassificationPointnet_10epochs_1024.pt"

    config = {"epochs": 10,
              "lr": 0.001,
              "batch_size": 32,
              "nPoints": 1024}   
    
    # Cuda agnostic thingy
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # SEEDS i histories varies
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Importing ModelNet
    # ModelNet dataset (original) stores objects as triangular meshes (.off files).
    # PointNet needs points, not meshes. "SamplePoints" (from torch_geometric) does that: 
    # samples N unifrom points from the object surface and gives you back a tensor
    transform = T.Compose([
        T.SamplePoints(config["nPoints"]),      # mesh -> point cloud (pos: [N,3])
        T.NormalizeScale(),                     # center + scale to unit sphere
    ])

    # DATASET + SPLIT
    # "pre_transform" is processed only once and saved to processed/*.pt forever (until deleted)
    # "transform" is processed every time __getitem__ is called, every time an object is called (i.e. it samples it)
    full_train_dataset = ModelNet(root="data/ModelNet", name="10", train=True, pre_transform=transform)
    test_dataset  = ModelNet(root="data/ModelNet", name="10", train=False, pre_transform=transform)
    # Split train_dataset into train and validation
    train_size = int(0.8 * len(full_train_dataset))
    val_size   = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)) # reproducibility)

    # DATALOADERS
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True) 
    val_loader   = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)  
    test_loader  = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # CHECKS: sizes
    print(f"Train: {len(train_dataset)}, num_classes: {train_dataset.dataset.num_classes}")
    print(f"Val  : {len(val_dataset)}, num_classes: {val_dataset.dataset.num_classes}")
    print(f"Test : {len(test_dataset)}, num_classes: {test_dataset.num_classes}")
    # NAME CLASSES
    # listdir: lists everything inside a directory
    # isdir: says if it is a directory
    path_folder = "data/ModelNet/raw"
    name_classes = sorted( d for d in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, d)))
    print("Name classes", name_classes) 
    info_dataset_batch(train_dataset, train_loader, name_classes)
    
    # MODEL + OPTIMIZER + LOSS
    network = ClassificationPointNet(num_classes=10).to(device)
    optimizer = optim.Adam(network.parameters(), lr=config["lr"])
    criterion = nn.NLLLoss()

    # TRAINING LOOP
    time_start_traning = time.time()
    train_loss, train_acc, val_loss, val_acc = train_model(config, train_loader, val_loader, network, optimizer, criterion, save_path=save_path)
    time_training = time.time() - time_start_traning
    print(f"Training time: {time_training}")

    # PLOT TRAINING AND ACCURACY CURVES
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2,1,1)
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title('Loss: training and validation')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='validation')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title('Accuracy: training and validation')
    plt.legend()
    plt.show()
    # plt.savefig("learning_curves.png")  # if server or remote instead of plt.show()

    # LATER
    # network = ClassificationPointNet(num_classes=10).to(device)
    # network.load_state_dict(torch.load(save_path, map_location=device))
    # network.eval()

    # FINAL TEST
#    test_loss, test_acc = eval_single_epoch(test_loader, trained_model, criterion)
#    print(f"Final test: loss={test_loss:.4f}, acc={test_acc:.2f}")

# %%
