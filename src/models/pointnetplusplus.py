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
      new_xyz: [B, S, 3]
      grouped: [B, S, K, 3 + D]  (if features is not None)
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
        knn_feat = gather_points_by_index(features, knn_idx)    # [B,S,K,D]
        grouped = torch.cat([knn_xyz_norm, knn_feat], dim=-1)   # [B,S,K,3+D]
    else:
        grouped = knn_xyz_norm                                  # [B,S,K,3]

    return centers_xyz, grouped



class PointNetSetAbstraction(nn.Module):
