import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


def square_distance(setA, setB):
    """  
    Computes pairwise squared distance between two sets of points from the same point cloud
    distance = ||a-b||^2 = ||a||^2 +||b||^2-2*a*b

    input: 
        setA: [B, S, 3]
        setB: [B, N, 3]

    returns: 
        distance: [B, S, N]    
    """
    setA2 = torch.sum(setA**2, dim=2, keepdim=True)                 # [B, S, 1]
    setB2 = torch.sum(setB**2, dim=2, keepdim=True).transpose(2,1)  # [B, 1, N]
    cross = torch.bmm(setA,setB.transpose(2,1))                     # [B, S, 3]*[B, 3, N]=[B, S, N]
    dist2 = setA2 + setB2 - 2*cross

    return dist2



def gather_points_by_index(points, idx):
    """
    Fetch points/features using indices: points[batch_idx, idx, :]

    input:
        points: [B, N, C]
        idx:    [B, S] or [B, S, K]

    returns:
        [B, S, C]     if idx is [B, S]
        [B, S, K, C]  if idx is [B, S, K]
    """

    device = points.device
    B = points.shape[0]

    if idx.dim() == 2:
        # idx shape: [B, S]
        batch_idx = torch.arange(B, device=device)[:, None]        # [B,1]
        new_points = points[batch_idx, idx, :]                     # [B,S,C]

    elif idx.dim() == 3:
        # idx shape: [B, S, K]
        batch_idx = torch.arange(B, device=device)[:, None, None]  # [B,1,1]
        new_points = points[batch_idx, idx, :]                     # [B,S,K,C]

    else:
        raise ValueError(f"idx must have shape [B,S] or [B,S,K], got {idx.shape}")

    return new_points





def knn_point(k, points, centers):
    """
    kNN search: 
    For each center point, find the indices of the k nearest points in the same cloud.
    
    inpùt:
        points:  [B, N, 3]
        centers: [B, S, 3]

    return:
        idx: [B,S,K] (indices of the K nearest points)
    """

    distance = square_distance(centers, points)                 # [B, S, N]
    _, idx = torch.topk(distance, k=k, dim=2, largest=False)

    return idx



def farthest_point_sample(points, num_centers):
    """
    We pick the point whose closest center is furthest
    For each point, compute the distance to the closest center. Then, once we have a collection 
    of minimum distances, we pick the furthest one (the one less represented by the centers). 

    Input:
        points:      point coordinates, shape [B, N, 3]
        num_centers: number of centers to sample  

    returns:
        fps_idx: indices of sampled centers, shape [B, S]
    """

    device = points.device
    B, N, _ = points.shape
    centers_idx = torch.zeros(B, num_centers, dtype=torch.long, device=device)          # [B,S]
    first_center_idx = torch.randint(0, N, (B,), device=device)       # [B]
    centers_idx[:,0] = first_center_idx
    
    # distance from first center to all points
    first_center_xyz = gather_points_by_index(points, first_center_idx.unsqueeze(1))    # [B,1,3] 
    distance = square_distance(first_center_xyz, points)                                # [B,1,N]
    # keep track of closest center
    distance_closest_center = distance.squeeze(1)                                       # [B,N] 

    for c in range(1, num_centers):
        # find point whose closest center is farthest
        new_center_idx = torch.argmax(distance_closest_center, dim=1)                   # [B]
        centers_idx[:,c] = new_center_idx                   
        
        # distance from new center to all points
        new_center_xyz = gather_points_by_index(points, new_center_idx.unsqueeze(1))    # [B,1,3]
        distance = square_distance(new_center_xyz, points).squeeze(1)                   # [B,N]

        # update closest center distance for all points
        distance_closest_center = torch.minimum(distance_closest_center, distance)

    return centers_idx  




def sample_and_group(num_centers, K, points, features=None):
    """
    S: number of centers to sample by FPS
    K: number of neighbors per center (kNN)
    points:      [B, N, 3]
    features: [B, N, D] or None

    returns:
      centers_xyz: [B, S, 3]
      new_knn_xyz: [B, S, K, 3 + D]  (if features is not None)
               [B, S, K, 3]      (otherwise)
    """    
    # Sample centers (indices)
    centers_idx = farthest_point_sample(points, num_centers)    # [B,S]
    
    # Gather center coordinates   
    centers_xyz = gather_points_by_index(points, centers_idx)   # [B,S,3]

    # Find K neighbors for each center
    knn_idx = knn_point(K, points=points, centers=centers_xyz)  # [B,S,K]

    # gather neighbor coordinates
    knn_xyz = gather_points_by_index(points, knn_idx)           # [B,S,K,3]

    # Normalize to local coordinates (centered at each center)
    knn_xyz_norm = knn_xyz - centers_xyz.unsqueeze(2)           # [B,S,K,3]

    if features is not None:
        knn_features = gather_points_by_index(features, knn_idx)    # [B,S,K,D]
        knn_points = torch.cat([knn_xyz_norm, knn_features], dim=-1)# [B,S,K,3+D]
    else:
        knn_points = knn_xyz_norm                               # [B,S,K,3]

    return centers_xyz, knn_points



class PointNetSetAbstraction(nn.Module):
    """
    Single-scale Set Abstraction (SA) layer using:
      - FPS to sample S centers
      - kNN to group K neighbors per center
      - shared MLP (Conv2d 1x1) + maxpool over neighbors
      to produce a feature per center.
    """    

    def __init__(self, num_centers, K, in_channels, mlp_channels):
        super().__init__()
        self.num_centers = num_centers
        self.K = K
        self.in_channels = in_channels
        self.mlp_channels = mlp_channels
        
        # Shared MLP over neighborhood points (implemented as 1x1 Conv2d)
        layers = []
        in_c = in_channels
        for out_c in mlp_channels:
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=1, bias=False))    
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            in_c = out_c
        # Sequential expects separated arguments, not a list. The * unpacks them   
        self.mlp = nn.Sequential(*layers)       


    def forward(self, points, features=None):
        """
        Input:
            points:   [B, N, 3]
            features: [B, N, D] or None
        Output:
            new_points:   [B, S, 3]        (the sampled centers xyz)
            new_features: [B, S, C_out]    (learned features per center)
        """
        # sample centers ans group neighbourhoods
        centers_xyz, knn_points = sample_and_group(self.num_centers, self.K, points, features=features)
        # centers_xyz: [B, S, 3]
        # knn_points:  [B, S, K, 3(+D)]
        
        # 2) Prepare for Conv2d: [B, S, K, C] -> [B, C, S, K]
        knn_points = knn_points.permute(0, 3, 1, 2).contiguous()      # [B, C_in, S, K]

        # 3) Local PointNet (shared MLP)
        x = self.mlp(knn_points)                                   # [B, C_out, S, K]

        # 4) Symmetric pooling over K neighbors
        x, _ = torch.max(x, dim=3)                              # [B, C_out, S]

        # 5) Back to [B, S, C_out]
        center_feat = x.permute(0, 2, 1).contiguous()          # [B, S, C_out]

        return centers_xyz, center_feat



class PointNetPlusPlusBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # SA1: no input features -> knn_points has 3 channels 
        self.sa1 = PointNetSetAbstraction(num_centers=512, K=32, in_channels=3, mlp_channels=[64, 64, 128])

        # SA2: features now 128 -> knn_points has 3 + 128
        self.sa2 = PointNetSetAbstraction(num_centers=128, K=32, in_channels=3 + 128, mlp_channels=[128, 128, 256])

        # SA3: features now 256 -> knn_points has 3 + 256
        self.sa3 = PointNetSetAbstraction(num_centers=32, K=32, in_channels=3 + 256, mlp_channels=[256, 512, 1024])

    def forward(self, xyz, features=None):
        """
        xyz:      [B, N, 3]
        features: [B, N, D] or None

        returns: levels = [(xyz0, f0), (xyz1, f1), (xyz2, f2), (xyz3, f3)]
        """
        xyz0, f0 = xyz, features
        xyz1, f1 = self.sa1(xyz0, features=f0)    # [B,512,3], [B,512,128]
        xyz2, f2 = self.sa2(xyz1, features=f1)    # [B,128,3], [B,128,256]
        xyz3, f3 = self.sa3(xyz2, features=f2)    # [B, 32,3], [B, 32,1024]

        return (xyz0,f0), (xyz1, f1), (xyz2, f2), (xyz3, f3)


class PointNetPlusPlusClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Backbone
        self.backbone = PointNetPlusPlusBackbone()
        
        # Classification head
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz, features=None):
        """
        xyz: [B, N, 3]
        features: [B, N, D] or None
        """
        _, _, _, (xyz3, f3) = self.backbone(xyz, features=features)   # features: [B,32,1024]

        # global pooling over centers dimension S=32
        global_features, _ = torch.max(f3, dim=1)     # [B,1024]

        x = self.dp1(F.relu(self.bn1(self.fc1(global_features))))
        x = self.dp2(F.relu(self.bn2(self.fc2(x))))
        logits = self.fc3(x)                                # [B,num_classes]
        log_probs = F.log_softmax(logits, dim=1)

        return log_probs
    



# RUN ONLY IF EXECUTED AS MAIN
if __name__ == "__main__":

    def set_device():
        if torch.cuda.is_available(): device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else: device = torch.device("cpu")

        print(f"\nUsing device: {device}")
        return device

    # GPU agnostic thingy
    device = set_device()

    import torch_geometric.transforms as T
    import torch.optim as optim
    from torch_geometric.datasets import ModelNet
    from torch.utils.data import random_split
    from torch_geometric.loader import DataLoader as DataLoaderGeometric
    from torch_geometric.utils import to_dense_batch
    from tqdm import tqdm

    # Importing ModelNet
    transform = T.Compose([T.SamplePoints(1024), T.NormalizeScale()])                   
    full_train_dataset = ModelNet(root="data/ModelNet", name="10", train=True, transform=transform)    
    test_dataset       = ModelNet(root="data/ModelNet", name="10", train=False, transform=transform)    
    
    train_size = int(0.8 * len(full_train_dataset))
    val_size   = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)) # reproducibility)

    # DATALOADERS
    train_loader = DataLoaderGeometric(train_dataset, batch_size=32, shuffle=True) 
    val_loader   = DataLoaderGeometric(val_dataset, batch_size=32, shuffle=False) 
    test_loader  = DataLoaderGeometric(test_dataset, batch_size=32, shuffle=False)

    network = PointNetPlusPlusClassifier(num_classes=10).to(device)
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    
    num_epochs = 20

    for epoch in range(num_epochs):

        # TRAINING
        network.train()
        total_loss, total_correct, total_seen = 0.0, 0, 0

        for batch in tqdm(train_loader, desc="train epoch", leave=False):

            batch = batch.to(device)

            # batch.pos: [B*N, 3]  -> [B, N, 3]
            xyz, _ = to_dense_batch(batch.pos, batch.batch)  
            labels = batch.y   
            B = labels.shape[0]

            optimizer.zero_grad()
            log_probs = network(xyz)                 # [B, num_classes]
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            preds = log_probs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_seen += B    
        
        train_loss = total_loss / total_seen
        train_acc  = total_correct / total_seen


        # VALIDATING
        network.eval()

        with torch.no_grad():

            total_loss, total_correct, total_seen = 0.0, 0, 0

            for batch in tqdm(val_loader, desc="val epoch", leave=False):
                batch = batch.to(device)
                # batch.pos: [B*N, 3]  -> [B, N, 3]
                xyz, _ = to_dense_batch(batch.pos, batch.batch)  
                labels = batch.y   
                B = labels.shape[0]

                log_probs = network(xyz)
                loss = criterion(log_probs, labels)

                total_loss += loss.item() * B
                preds = log_probs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_seen += B

            val_loss = total_loss / total_seen
            val_acc  = total_correct / total_seen

        tqdm.write(f"Epoch: {epoch+1}/{num_epochs}"
            f" | loss (train/val) = {train_loss:.3f}/{val_loss:.3f}"
            f" | acc (train/val) = {train_acc:.3f}/{val_acc:.3f}")