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
from pointNet import ClassificationPointNet, BasePointNet, TransformationNet
import random
import os

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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)  
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %% CHECKS: sizes, inspect batch
print(f"Train: {len(train_dataset)}, num_classes: {train_dataset.dataset.num_classes}")
print(f"Val  : {len(val_dataset)}, num_classes: {val_dataset.dataset.num_classes}")
print(f"Test : {len(test_dataset)}, num_classes: {test_dataset.num_classes}")
# classes names
folders = sorted(os.listdir("data/ModelNet/raw"))
print("Name classes", folders) 

batch = next(iter(train_loader))
print(batch)
if 1:
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


# %%
# ----------------------------------------------------
#    TRAINING EPOCH FUNCTION (iterate on data_loader)
# ----------------------------------------------------
def train_single_epoch(train_loader, network, optimizer, criterion):
    
    network.train()                             # Activate the train=True flag inside the model

    loss_history = []
    nCorrect = 0
    nTotal = 0
    for batch in train_loader:

        batch = batch.to(device)
        label = batch.y                                                 
        # Pointnet needs: [object, nPoints, coordinades] 
        # i.e. [32 object, 1024 points, 3 coordinates]: [batch_size, nPoints, 3]
        # Since batch.batch is [32x1024, 3] we have to split it into individual object points.
        # It could be done manually but "to_dense_batch" does that.
        # "to_dense_batch" will also add padding if not all objects have same number of points. In our case they have.
        # mask is a [batch_size, nPoints] boolean saying if an entry is actually a real point or padding
        BatchPointsCoords, _ = to_dense_batch(batch.pos, batch.batch)   

        optimizer.zero_grad()                  # Set network gradients to 0
        log_probs, _, _, feature_tnet_tensor, _ = network(BatchPointsCoords)    # Forward batch through the network      
        # REGULARIZATION: force Tnet matrix to be orthogonal (TT^t = I)
        # i.e. allow transforming the sapce but without distorting it
        # The loss adds this term to be minimized: ||I-TT^t||
        # It is a training constrain --> no need to be included in validation
        TT = torch.bmm(feature_tnet_tensor, feature_tnet_tensor.transpose(2, 1))
        I = torch.eye(TT.shape[-1], device=TT.device).unsqueeze(0).expand(TT.shape[0], -1, -1) # [64,64]->[1,64,64]->[batch,64,64]
        reg_loss = torch.norm(I - TT) / TT.shape[0]                 # make reg_loss batch invariant (dividing by batch_size)
        loss = criterion(log_probs, label) + 0.001 * reg_loss       # Compute loss: NLLLoss   
        loss.backward()                                             # Compute backpropagation
        optimizer.step()    
        # Compute metrics
        loss_history.append(loss.item())         
        prediction = log_probs.argmax(dim=1)
        batch_correct = (prediction == label).sum().item()          # .item() brings one single scalar to CPU
        nCorrect = nCorrect + batch_correct
        nTotal = nTotal + len(label)
        
    # Average across all batches    
    train_loss_epoch = np.mean(loss_history) 
    train_acc_epoch = nCorrect / nTotal
    
    return train_loss_epoch, train_acc_epoch
# ----------------------------------------------------


# ----------------------------------------------------
#    TESTING EPOCH FUNCTION (iterate on data_loader)
# ----------------------------------------------------
def eval_single_epoch(data_loader, network, criterion):

    with torch.no_grad():
        network.eval()                      # Dectivate the train=True flag inside the model

        loss_history = []
        nCorrect = 0
        nTotal = 0
        for batch in data_loader:
            batch = batch.to(device)

            label = batch.y 
            BatchPointsCoords, _ = to_dense_batch(batch.pos, batch.batch)   

            log_probs, _, _, _, _ = network(BatchPointsCoords)  # Forward batch through the network
            loss = criterion(log_probs, label)                  # Compute loss
            # Compute metrics
            loss_history.append(loss.item())         
            prediction = log_probs.argmax(dim=1)
            batch_correct = (prediction == label).sum().item()  # .item() brings one single scalar to CPU
            nCorrect = nCorrect + batch_correct
            nTotal = nTotal + len(label)
        
        # Average across all batches 
        eval_loss_epoch = np.mean(loss_history)       
        eval_acc_epoch = nCorrect / nTotal
    
    return eval_loss_epoch, eval_acc_epoch
# ----------------------------------------------------

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
              "batch_size": 100}   

    # MODEL + OPTIMIZER + LOSS
    network = ClassificationPointNet(num_classes=10).to(device)
    optimizer = optim.Adam(network.parameters(), lr=config["lr"])
    criterion = nn.NLLLoss()

    # TRAINING LOOP
    trained_network = train_model(config, train_loader, val_loader, network, optimizer, criterion)

    plt.show()

    # FINAL TEST
#    test_loss, test_acc = eval_single_epoch(test_loader, trained_model, criterion)
#    print(f"Final test: loss={test_loss:.4f}, acc={test_acc:.2f}")
