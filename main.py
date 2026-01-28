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
from training_utils import train_single_epoch, eval_single_epoch

# ----------------------------------------------------
#    TRAINING LOOP (iterate on epochs)
# ----------------------------------------------------
def train_model(config, train_loader, val_loader, network, optimizer, criterion):

    train_loss=[]
    train_acc=[]
    val_loss=[]
    val_acc=[]

    for epoch in range(config["epochs"]):
        train_loss_epoch, train_acc_epoch = train_single_epoch(train_loader, network, optimizer, criterion)
        val_loss_epoch, val_acc_epoch = eval_single_epoch(val_loader, network, criterion)
        
        train_loss.append(train_loss_epoch)
        train_acc.append(train_acc_epoch)
        val_loss.append(val_loss_epoch)
        val_acc.append(val_acc_epoch)

        print(f'Epoch: {epoch+1}/{config["epochs"]} | loss (train/val) = {train_loss_epoch:.4f}/{val_loss_epoch:.4f} | acc (train/val) ={train_acc_epoch:.2f}/{val_acc_epoch:.2f}')
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2,1,1)
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='val')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # plt.savefig("learning_curves.png")  # if server or remote instead of plt.show()

    return network


# RUN ONLY IF EXECUTED AS MAIN
if __name__ == "__main__":

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

    # TRANSFORMS
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

    # CHECKS: sizes, inspect batch
    print(f"Train: {len(train_dataset)}, num_classes: {train_dataset.dataset.num_classes}")
    print(f"Val  : {len(val_dataset)}, num_classes: {val_dataset.dataset.num_classes}")
    print(f"Test : {len(test_dataset)}, num_classes: {test_dataset.num_classes}")
    # classes names
    folders = sorted(os.listdir("data/ModelNet/raw"))
    print("Name classes", folders) 

    batch = next(iter(train_loader))
    print(batch)
    if 0:
        # There are 32 objects per batch, and each object has 1024 points. 
        # But it does not separate objects....all 32x1024 points are stacked together.
        # batch is a Data object (like a struct in matlab), with attributes. The atributes are:
        # batch.pos   --> llista de punts per batch ([32x1024,3])
        # batch.batch --> integer telling each point to which object it belongs to.
        # batch.y     --> labels, one label per object (not point!!!)
        # batch.ptr   --> ni puta idea
        # They can also be accessed like a dictionary: batch["pos"], batch["y"]...

        # info item from a batch
        batchItem = 30
        pointsObjectID = batch.pos[batch.batch==batchItem]
        print("Item batch shape:", pointsObjectID.shape, "Label:", batch.y[batchItem])

        # info item from the original dataset
        datasetItem = 1000
        pos = full_train_dataset[datasetItem].pos.cpu().numpy()
        label = full_train_dataset[datasetItem].y.item()
        fig = plt.figure(figsize=[7,7])
        ax = plt.axes(projection='3d')
        sc = ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=pos[:,0] ,s=80, marker='o', cmap="viridis", alpha=0.7)
        ax.set_zlim3d(-1, 1)
        plt.title(f'Label: {label}')
        plt.show()


    # MODEL + OPTIMIZER + LOSS
    network = ClassificationPointNet(num_classes=10).to(device)
    optimizer = optim.Adam(network.parameters(), lr=config["lr"])
    criterion = nn.NLLLoss()

    # TRAINING LOOP
    trained_network = train_model(config, train_loader, val_loader, network, optimizer, criterion)

    plt.show()

    save_path = "checkpoints/ClassificationPointnet_10epochs_1024.pt"
    torch.save({"model": network.state_dict(),
                "epochs": config["epochs"], 
                "nPoints": config["nPoints"]},
                save_path)
    # LATER
    # network = ClassificationPointNet(num_classes=10).to(device)
    # network.load_state_dict(torch.load(save_path, map_location=device))
    # network.eval()

    # FINAL TEST
#    test_loss, test_acc = eval_single_epoch(test_loader, trained_model, criterion)
#    print(f"Final test: loss={test_loss:.4f}, acc={test_acc:.2f}")

# %%
