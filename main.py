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
from pointNet import PointNetSimple
import random


# Cuda agnostic thingy
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# SEEDS i histories varies
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# %% Importing ModelNet
# ModelNet dataset (original) stores objects as triangular meshes (.off files).
# PointNet needs points, not meshes. 
# "SamplePoints" (from torch_geometric) does that: samples N unifrom points from the object surface
# and gives you back a tensor
nPoints = 1024

# TRANSFORMS
transform = T.Compose([
    T.SamplePoints(nPoints),        # mesh -> point cloud (pos: [N,3])
    T.NormalizeScale(),             # center + scale to unit sphere
])

# DATASET + SPLIT
full_train_dataset = ModelNet(root="data/ModelNet", name="10", train=True, transform=transform)
test_dataset  = ModelNet(root="data/ModelNet", name="10", train=False, transform=transform)
# Split train_dataset into train and validation
train_size = int(0.8 * len(full_train_dataset))
val_size   = len(full_train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)) # reproducibility)

# DATALOADERS
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)#, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)#, num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)#, num_workers=4)

# %% CHECKS: sizez, inspect batch
print("Train:", len(train_dataset))
print("Val  :", len(val_dataset))
print("Test :", len(test_dataset))
# There are 32 objects per batch, and each object has 1024 points. 
# But it does not separate objects....all 32x1024 points are stacked together.
# batch is a Data object (like a struct in matlab), with attributes. 
# The atributes are:
# batch.pos   --> llista de punts per batch ([32x1024,3])
# batch.batch --> integer telling each point to which object it belongs to.
# batch.y     --> labels, one label per object (not point!!!)
# batch.ptr   --> ni puta idea
# They can also be accessed like a dictionary: batch["pos"], batch["y"]...

batch = next(iter(train_loader))
print(batch)

objectID = 1
pointsObjectID = batch.pos[batch.batch==objectID]
print(pointsObjectID.shape)  
print("Label:", batch.y[objectID])

if 0:
    pos = full_train_dataset[1].pos.numpy()
    label = full_train_dataset[1].y.item()
    fig = plt.figure(figsize=[7,7])
    ax = plt.axes(projection='3d')
    sc = ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=pos[:,0] ,s=80, marker='o', cmap="viridis", alpha=0.7)
    ax.set_zlim3d(-1, 1)
    plt.title(f'Label: {label}')

# %%
# ----------------------------------------------------
#    TRAINING EPOCH FUNCTION (iterate on data_loader)
# ----------------------------------------------------
def train_single_epoch(train_loader, network, optimizer, criterion):
    
    network.train()                             # Activate the train=True flag inside the model

    loss_history = []
    acc_history = []
    for batch in train_loader:

        batch = batch.to(device)
        # Pointnet needs: [object, nPoints, coordinades]
        #  i.e. [32 object, 1024 points, 3 coordinates]: [batch_size, nPoints, 3]
        # to_dense_batch will do padding if not all objects have same number of points. In our case they have.
        # mask is a [batch_size, nPoints] boolean saying if an entry is actually a real point or padding
        BatchPointsCoords, _ = to_dense_batch(batch.pos, batch.batch)   
        label = batch.y                                                 

        optimizer.zero_grad()                  # Set network gradients to 0
        output = network(BatchPointsCoords)    # Forward batch through the network
        loss = criterion(output, label)        # Compute loss
        loss.backward()                        # Compute backpropagation
        optimizer.step()    
        # Compute metrics
        loss_history.append(loss.item())         
        acc_history.append(accuracy(label, output))
        
    # Average across all batches    
    train_loss = np.mean(loss_history) 
    train_acc = np.mean(acc_history) 
    
    return train_loss, train_acc
# ----------------------------------------------------


# ----------------------------------------------------
#    TESTING EPOCH FUNCTION (iterate on data_loader)
# ----------------------------------------------------
def eval_single_epoch(data_loader, network, criterion):

    with torch.no_grad():
        network.eval()                      # Dectivate the train=True flag inside the model

        loss_history = []
        acc_history = []
        for batch in data_loader:
            batch = batch.to(device)

            BatchPointsCoords, _ = to_dense_batch(batch.pos, batch.batch)   
            label = batch.y 

            output = network(BatchPointsCoords)       # Forward batch through the network
            loss = criterion(output, label)           # Compute loss
            # Compute metrics
            loss_history.append(loss.item())         
            acc_history.append(accuracy(label, output))
        
        # Average across all batches 
        eval_loss = np.mean(loss_history)       
        eval_acc = np.mean(acc_history) 
    
    return eval_loss, eval_acc
# ----------------------------------------------------

# ----------------------------------------------------
#    TRAINING LOOP (iterate on epochs)
# ----------------------------------------------------
def train_model(config, train_loader, val_loader, model, optimizer, criterion):

    train_loss_epoch=[]
    train_acc_epoch=[]
    val_loss_epoch=[]
    val_acc_epoch=[]

    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_single_epoch(train_loader, model, optimizer, criterion)
        val_loss, val_acc = eval_single_epoch(val_loader, model, criterion)
        
        train_loss_epoch.append(train_loss)
        train_acc_epoch.append(train_acc)
        val_loss_epoch.append(val_loss)
        val_acc_epoch.append(val_acc)

        print(f'Epoch: {epoch+1}/{config["epochs"]} | loss (train/val) = {train_loss:.4f}/{val_loss:.4f} | acc (train/val) ={train_acc:.2f}/{val_acc:.2f}')
    
    return model


# RUN ONLY IF EXECUTED AS MAIN
if __name__ == "__main__":

    config = {"epochs": 10,
              "lr": 0.001,
              "batch_size": 100}   

    # MODEL + OPTIMIZER + LOSS
    model = PointNetSimple(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    # TRAINING LOOP
    trained_model = train_model(config, train_loader, val_loader, model, optimizer, criterion)

    # FINAL TEST
    test_loss, test_acc = eval_single_epoch(test_loader, trained_model, criterion)
    print(f"Final test: loss={test_loss:.4f}, acc={test_acc:.2f}")
