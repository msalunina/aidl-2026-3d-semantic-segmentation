import torch
import torch.nn as nn
import torch.nn.functional as F


def gather_points_by_index(points, idx):
    """
    Fetch points/features using indices: points[batch_idx, idx, :]

    Input:
        points: [B, N, C]
        idx:    [B, S] or [B, S, K]

    Output:
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



def square_distance(setA, setB):
    """  
    Compute pairwise squared Euclidean distances between two sets of points.
    distance = ||a-b||^2 = ||a||^2 +||b||^2-2*a*b

    Input: 
        setA: [B, S, 3]
        setB: [B, N, 3]

    Output: 
        dist2: [B, S, N]    
    """
    setA2 = torch.sum(setA**2, dim=2, keepdim=True)                 # [B, S, 1]
    setB2 = torch.sum(setB**2, dim=2, keepdim=True).transpose(2,1)  # [B, 1, N]
    cross = torch.bmm(setA,setB.transpose(2,1))                     # [B, S, 3]*[B, 3, N]=[B, S, N]
    dist2 = setA2 + setB2 - 2*cross
    dist2 = torch.clamp(dist2, min=0.0)                             # avoid tiny negative

    return dist2



def farthest_point_sample(points_xyz, num_centers):
    """
    We pick the point whose closest center is furthest
    For each point, compute the distance to the closest center. Then, once we have a collection 
    of minimum distances, we pick the furthest one (the one less represented by the centers). 

    Input:
        points_xyz:  point coordinates, shape [B, N, 3]
        num_centers: number of centers to sample  

    Output:
        fps_idx: indices of sampled centers, shape [B, S]
    """

    device = points_xyz.device
    B, N, _ = points_xyz.shape
    centers_idx = torch.zeros(B, num_centers, dtype=torch.long, device=device)              # [B,S]
    first_center_idx = torch.randint(0, N, (B,), device=device)                             # [B]
    centers_idx[:,0] = first_center_idx
    
    # distance from first center to all points
    first_center_xyz = gather_points_by_index(points_xyz, first_center_idx.unsqueeze(1))    # [B,1,3] 
    distance = square_distance(first_center_xyz, points_xyz)                                # [B,1,N]
    # keep track of closest center
    distance_closest_center = distance.squeeze(1)                                           # [B,N] 

    for c in range(1, num_centers):
        # find point whose closest center is farthest
        new_center_idx = torch.argmax(distance_closest_center, dim=1)                       # [B]
        centers_idx[:,c] = new_center_idx                   
        
        # distance from new center to all points
        new_center_xyz = gather_points_by_index(points_xyz, new_center_idx.unsqueeze(1))    # [B,1,3]
        distance = square_distance(new_center_xyz, points_xyz).squeeze(1)                   # [B,N]

        # update closest center distance for all points
        distance_closest_center = torch.minimum(distance_closest_center, distance)

    return centers_idx  



def knn_point(k, reference_points, query_points):
    """
    kNN search: 
    For each query point (center), find the indices of the k nearest points in the same cloud (reference_points)
    
    Input:
        reference_points:  [B, N, 3]     (search space)
        query_points:      [B, S, 3]     (points asking for neighbors)

    Output:
        idx: [B,S,K] (indices of the K nearest points)
    """

    distance2 = square_distance(query_points, reference_points)                 # [B, S, N]
    _, knn_idx = torch.topk(distance2, k=k, dim=2, largest=False, sorted=False)

    return knn_idx




def query_ball_point_closest(radius, K, reference_points, query_points):
    """
    For each query point (center), find up to K reference points
    lying inside a ball of radius 'radius'.

    Assumption: query_points are sampled from reference_points,so 
    each center is itself one of the reference points. Therefore 
    distance(center, center) = 0 <= radius, so each neighborhood 
    contains at least one valid point.

    Input:
        radius:            radius of the ball
        K:                 maximum number of neighbors per center
        reference_points:  [B, N, 3]
        query_points:      [B, S, 3]

    Output:
        group_idx: [B, S, K]
    """
    if radius is None:
        raise ValueError("radius must be provided for ball query")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    # Squared distances from each center to all points in the cloud
    distance2 = square_distance(query_points, reference_points)             # [B,S,N]
    
    # Points outside the radius are assigned infinite distance
    distance2[distance2 > radius ** 2] = torch.inf                          # [B,S,N]

    # Take K smallest distances = K closest valid neighbors
    selected_distance2, group_idx = torch.topk(distance2, K, dim=-1, largest=False) # both [B, S, K]

    # Replace invalid entries with the closest valid neighbor
    first_valid_idx = group_idx[:, :, 0].unsqueeze(-1).expand(-1, -1, K)    # [B,S]->[B,S,1]->[B,S,K]
    invalid_mask = torch.isinf(selected_distance2)                          # [B,S,K]
    group_idx[invalid_mask] = first_valid_idx[invalid_mask]                 # [B,S,K]

    return group_idx



def query_ball_point_random(radius, K, reference_points, query_points):
    """
    For each query point (center), find up to K reference points
    lying inside a ball of radius 'radius'.

    If more than K points fall inside the ball, K of them are randomly chosen.

    Assumption: query_points are sampled from reference_points, so 
    each center is itself one of the reference points. Therefore 
    distance(center, center) = 0 <= radius, so each neighborhood 
    contains at least one valid point.

    Input:
        radius:            radius of the ball
        K:                 maximum number of neighbors per center
        reference_points:  [B, N, 3]
        query_points:      [B, S, 3]

    Output:
        group_idx: [B, S, K]
    """

    if radius is None:
        raise ValueError("radius must be provided for ball query")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    device = reference_points.device
    B, S, _ = query_points.shape
    N = reference_points.shape[1]

    # Squared distances from each center to all points in the cloud
    distance2 = square_distance(query_points, reference_points)                     # [B,S,N]

    # Assign a random score between 0 and 1 to each candidate neighbor
    # Change score to inf to those points outside the radius, so they are 
    # pushed to the end when choosing topk
    random_score = torch.rand(B, S, N, device=device)                               # [B,S,N]
    random_score[distance2 > radius ** 2] = torch.inf                               # [B,S,N]

    # Take K smallest scores = K random valid neighbors
    selected_score, group_idx = torch.topk(random_score, K, dim=-1, largest=False)  # [B,S,K]

    # Replace invalid entries with the first valid neighbor
    first_valid_idx = group_idx[:, :, 0].unsqueeze(-1).expand(-1, -1, K)            # [B,S,K]
    invalid_mask = torch.isinf(selected_score)                                      # [B,S,K]
    group_idx[invalid_mask] = first_valid_idx[invalid_mask]

    return group_idx





def sample_and_group(num_centers, K, points_xyz, points_features=None, grouping="knn", radius=None):
    """
    num_centers: number of centers to sample by FPS
    K:           number of neighbors per center (kNN), or maximum number
                 of neighbors per center (ball query)

    Input:
        points_xyz:      [B, N, 3]
        points_features: [B, N, D] or None
        grouping:        "knn" or "ball"
        radius:          ball radius (required if grouping == "ball")

    Output:
      centers_xyz: [B, S, 3]
      group_data:  [B, S, K, 3 + D]  (if features is not None) (geometry + features)
                   [B, S, K, 3]      (otherwise)
    """
    # Sample centers (indices)
    centers_idx = farthest_point_sample(points_xyz, num_centers)                      # [B,S]

    # Gather center coordinates
    centers_xyz = gather_points_by_index(points_xyz, centers_idx)                     # [B,S,3]

    # Find neighbors for each center
    if grouping == "knn":
        group_idx = knn_point(K, reference_points=points_xyz, query_points=centers_xyz)   # [B,S,K]

    elif grouping == "ball_closest":
        group_idx = query_ball_point_closest(radius=radius,K=K,reference_points=points_xyz,query_points=centers_xyz) # [B,S,K]
    
    elif grouping == "ball_random":
        group_idx = query_ball_point_random(radius=radius,K=K,reference_points=points_xyz,query_points=centers_xyz) # [B,S,K]

    else:
        raise ValueError(f"Unknown grouping mode '{grouping}'. Use 'knn', 'ball_closest' or 'ball_random'.")

    # Gather neighbor coordinates
    group_xyz = gather_points_by_index(points_xyz, group_idx)                         # [B,S,K,3]

    # Normalize to local coordinates (centered at each center)
    group_xyz_norm = group_xyz - centers_xyz.unsqueeze(2)                             # [B,S,K,3]

    if points_features is not None:
        group_features = gather_points_by_index(points_features, group_idx)           # [B,S,K,D]
        group_data = torch.cat([group_xyz_norm, group_features], dim=-1)              # [B,S,K,3+D]
    else:
        group_data = group_xyz_norm                                                   # [B,S,K,3]

    return centers_xyz, group_data




def interpolate_features_3nn(target_xyz, source_xyz, source_features, eps=1e-8):
    """
    Interpolate features from source points to target points using 3-NN.

    Input:
        target_xyz:      [B, N, 3]   (points where we want features)
        source_xyz:      [B, S, 3]   (points that already have features)
        source_features: [B, S, C]

    Output:
        target_features: [B, N, C]
    """
    # 1) find 3 nearest source points for each target point
    idx = knn_point(k=3, reference_points=source_xyz, query_points=target_xyz)  # [B, N, k]
    nearest_xyz = gather_points_by_index(source_xyz, idx)                       # [B, N, k, 3]

    # 2) compute distances: target - neighbour
    diff = target_xyz.unsqueeze(2) - nearest_xyz                                # [B, N, k, 3]
    dist2 = torch.sum(diff * diff, dim=-1)                                      # [B, N, k]

    # 3) convert to inverse-distance weights and normalize them: w1+w2+w3=1
    w = 1.0 / (dist2 + eps)                                         # [B, N, k]
    w = w / torch.sum(w, dim=2, keepdim=True)                                   # [B, N, k]

    # 4) Weighted sum of the 3 neighbor features
    nearest_features = gather_points_by_index(source_features, idx)             # [B, N, k, C]
    target_features = torch.sum(nearest_features * w.unsqueeze(-1), dim=2)      # [B, N, C]

    return target_features


# ----------------------------------------------------
#             SET ABSTRACTION 
# ----------------------------------------------------
class PointNetSetAbstraction(nn.Module):
    """
    Single-scale Set Abstraction (SA) layer using:
      - FPS to sample S centers
      - local grouping around each center (kNN or ball query)
      - shared MLP (Conv2d 1x1) + maxpool over neighbors to produce a feature per center.
    """    

    def __init__(self, num_centers, K, in_channels, mlp_channels, grouping="knn", radius=None):
        super().__init__()
        self.num_centers = num_centers
        self.K = K
        self.in_channels = in_channels
        self.mlp_channels = mlp_channels
        self.grouping = grouping
        self.radius = radius
        
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


    def forward(self, xyz, features=None):
        """
        Input:
            xyz:      [B, N, 3]
            features: [B, N, D] or None
        Output:
            centers_xyz:      [B, S, 3]        (the sampled centers xyz)
            centers_features: [B, S, C_out]    (learned features per center)
        """
        # sample centers and group neighbourhoods
        centers_xyz, group_data = sample_and_group(self.num_centers,
                                                   self.K,
                                                   points_xyz=xyz,
                                                   points_features=features,
                                                   grouping=self.grouping,
                                                   radius=self.radius)
        # centers_xyz: [B, S, 3]
        # group_data:  [B, S, K, 3(+D)]
        
        # 2) Permute for Conv2d: [B, S, K, C] -> [B, C, S, K]
        group_data = group_data.permute(0, 3, 1, 2).contiguous()    # [B, C_in, S, K]

        # 3) Local PointNet (shared MLP)
        x = self.mlp(group_data)                                  # [B, C_out, S, K]

        # 4) Symmetric pooling over K neighbors
        x, _ = torch.max(x, dim=3)                              # [B, C_out, S]

        # 5) Back to [B, S, C_out]
        centers_features = x.permute(0, 2, 1).contiguous()      # [B, S, C_out]

        return centers_xyz, centers_features
    


class PointNetSetAbstractionGlobal(nn.Module):
    """
    Global Set Abstraction layer.

    Applies a PointNet (shared MLP + maxpool) over the whole point set
    to produce a single global feature.
    """

    def __init__(self, in_channels, mlp_channels):
        super().__init__()

        layers = []
        in_c = in_channels
        for out_c in mlp_channels:
            layers.append(nn.Conv1d(in_c, out_c, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm1d(out_c))
            layers.append(nn.ReLU(inplace=True))
            in_c = out_c

        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, features=None):
        """
        Input:
            xyz:      [B, N, 3]
            features: [B, N, D] or None

        Output:
            global_features: [B, C_out]
        """

        if features is not None:
            points_data = torch.cat([xyz, features], dim=-1)                # [B,N,3+D]
        else:
            points_data = xyz                                               # [B,N,3]

        points_data = points_data.permute(0, 2, 1).contiguous()             # [B,C,N]
        points_data = self.mlp(points_data)                                 # [B,C_out,N]
        global_features, _ = torch.max(points_data, dim=2, keepdim=False)   # [B,C_out]

        return global_features


# ----------------------------------------------------
#             FEATURE PROPAGATION 
# ----------------------------------------------------
class PointNetFeaturePropagation(nn.Module):
    """
    PointNet++ Feature Propagation layer.
    Interpolates features from a sparse set (source) to a denser set (target),
    concatenates with skip features, then applies a shared MLP.
    """

    def __init__(self, in_channels, mlp_channels):
        """
        in_channels: channels after concatenation (C_interp + C_skip)
        mlp_channels: list like [256, 128] producing the final C_out=128
        """
        super().__init__()
        self.in_channels = in_channels
        self.mlp_channels = mlp_channels

        # Shared MLP applied on points (implemented as 1x1 Conv1d)
        layers = []
        in_c = in_channels
        for out_c in mlp_channels:
            layers.append(nn.Conv1d(in_c, out_c, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm1d(out_c))
            layers.append(nn.ReLU(inplace=True))
            in_c = out_c
        # Sequential expects separated arguments, not a list. The * unpacks them   
        self.mlp = nn.Sequential(*layers)

    def forward(self, target_xyz, source_xyz, source_features, target_skip_features=None):
        """
        Input:
            target_xyz:           [B, N, 3]   (denser)
            source_xyz:           [B, S, 3]   (sparser)
            source_features:      [B, S, C2]  (features at source points)
            target_skip_features: [B, N, C1] or None

        Output:
            new_target_features: [B, N, C_out]
        """
        # 1) interpolate source features onto target points
        target_features = interpolate_features_3nn(target_xyz=target_xyz, 
                                                   source_xyz=source_xyz, 
                                                   source_features=source_features)             # [B, N, C2]

        # 2) concat skip (if available)
        if target_skip_features is not None:
            joined_target_features = torch.cat([target_features, target_skip_features], dim=-1) # [B, N, C2+C1]
        else:
            joined_target_features = target_features                                            # [B, N, C2]

        # 3) shared MLP expects [B, C, N]
        joined_target_features = joined_target_features.permute(0, 2, 1).contiguous()           # [B, C_in, N]
        new_target_features = self.mlp(joined_target_features)                                  # [B, C_out, N]
        new_target_features = new_target_features.permute(0, 2, 1).contiguous()                 # [B, N, C_out]
        
        return new_target_features



# ----------------------------------------------------
#           POINTNET++ CLASSIFICATION
# ----------------------------------------------------
class PointNetPlusPlusClassifier(nn.Module):  
    """
    PointNet++ classifier (SSG architecture from the paper).

    Encoder:
    SA(512, [64,64,128])
    SA(128, [128,128,256])
    Global SA([256,512,1024])

    Head: FC(512) → FC(256) → FC(num_classes)
    """

    def __init__(self, num_classes, extra_channels=0, dropout = 0.5):
        """
        num_classes:     number of semantic classes
        extra_channels:  D (extra input features per point besides xyz). If you only have xyz -> 0.
        dropout:         dropout in the final classifier head
        """
        super().__init__()
        self.num_classes = num_classes
        self.extra_channels = extra_channels

        # -----------------------
        #       Encoder 
        # -----------------------
        # input:  xyz=[B,N,3],   feat=[B,N,C-3] or None       
        # output: xyz=[B,512,3], feat=[B,512,128]
        self.sa1 = PointNetSetAbstraction(num_centers=512, 
                                          K=32, 
                                          in_channels=3 + extra_channels, 
                                          mlp_channels=[64, 64, 128], 
                                          grouping = "knn")
        # input:  xyz=[B,512,3], feat=[B,512,128] 
        # output: xyz=[B,128,3], feat=[B,128,256] 
        self.sa2 = PointNetSetAbstraction(num_centers=128, 
                                          K=32, 
                                          in_channels=3 + 128, 
                                          mlp_channels=[128, 128, 256],
                                          grouping = "knn")
        # input:  xyz=[B,128,3], feat=[B,128,256] 
        # output: global_features=[B,1024] 
        self.sa3 = PointNetSetAbstractionGlobal(in_channels=3 + 256, mlp_channels=[256, 512, 1024])

        # -----------------------       
        #   Classification head
        # -----------------------
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(256, num_classes)
        )

    def forward(self, points):
        """
        Input:
            points: [B, N, C]  (xyz + optional features)
        Ouput:
            log_probs: [B, num_classes]
        """
        assert points.ndim == 3, f"Expected [B,N,C] tensor, got shape {points.shape}"
        assert points.shape[2] == 3 + self.extra_channels, \
            f"Expected {3 + self.extra_channels} channels (xyz + {self.extra_channels}), got {points.shape[2]}"

        xyz0 = points[:, :, :3]

        if points.shape[2] > 3:
            feat0 = points[:, :, 3:]
        else:
            feat0 = None

        # -------- Encoder --------
        xyz1, feat1 = self.sa1(xyz0, feat0)             # xyz1 [B,512,3], feat1 [B,512,128]
        xyz2, feat2 = self.sa2(xyz1, feat1)             # xyz2 [B,128,3], feat2 [B,128,256]
        global_features = self.sa3(xyz2, feat2)         # global_features [B,1024]     
        
        # -------- Head --------
        logits = self.classifier(global_features)       # [B,num_classes]
        log_probs = F.log_softmax(logits, dim=1)

        return log_probs
    



# ----------------------------------------------------
#               POINTNET++ SEGMENTATION
# ----------------------------------------------------
class PointNetPlusPlusSegmentation(nn.Module):
    """
    PointNet++ semantic segmentation (single-scale grouping version).
    """

    def __init__(self, num_classes, extra_channels = 0, dropout = 0.5, grouping="knn", K = None, radius=None):
        """
        num_classes:     number of semantic classes
        extra_channels:  D (extra input features per point besides xyz). If you only have xyz -> 0.
        dropout:         dropout in the final classifier head
        """
        super().__init__()
        self.num_classes = num_classes
        self.extra_channels = extra_channels
        self.grouping = grouping

        # Checks for radius
        if radius is None:
            radius = [None, None, None, None]

        if len(radius) != 4:
            raise ValueError(f"Expected 4 radius values, got {len(radius)}")
        
        # Checks for K
        if K is None:
            K = [32, 32, 32, 32]

        if len(K) != 4:
            raise ValueError(f"Expected 4 K values, got {len(K)}")
        
        self.K = K
        self.radius = radius

        # -----------------------
        #        Encoder 
        # -----------------------
        # input:  xyz=[B,N,3],    feat=[B,N,C-3] or None
        # output: xyz=[B,1024,3], feat=[B,1024,64]
        self.sa1 = PointNetSetAbstraction(num_centers=1024, 
                                          K=self.K[0], 
                                          in_channels=3 + extra_channels, 
                                          mlp_channels=[32, 32, 64],
                                          grouping=self.grouping,
                                          radius=self.radius[0])                  
        # input:  xyz=[B,1024,3], feat=[B,1024,64]
        # output: xyz=[B,256,3],  feat=[B,256,128]
        self.sa2 = PointNetSetAbstraction(num_centers=256, 
                                          K=self.K[1], 
                                          in_channels=3 + 64, 
                                          mlp_channels=[64, 64, 128],
                                          grouping=self.grouping,
                                          radius=self.radius[1]) 
        # input:  xyz=[B,256,3], feat=[B,256,128]
        # output: xyz=[B,64,3], feat=[B,64,256]
        self.sa3 = PointNetSetAbstraction(num_centers=64, 
                                          K=self.K[2], 
                                          in_channels=3 + 128, 
                                          mlp_channels=[128, 128, 256],
                                          grouping=self.grouping,
                                          radius=self.radius[2]) 
        # input:  xyz=[B,64,3], feat=[B,64,256]
        # output: xyz=[B,16,3], feat=[B,16,512]
        self.sa4 = PointNetSetAbstraction(num_centers=16, 
                                          K=self.K[3], 
                                          in_channels=3 + 256, 
                                          mlp_channels=[256, 256, 512],
                                          grouping=self.grouping,
                                          radius=self.radius[3]) 
       
        # -----------------------
        #        Decoder 
        # -----------------------
        # 16 -> 64 : interpolated features gives 512 + skip has 256 = 768
        self.fp4 = PointNetFeaturePropagation(in_channels=512 + 256, mlp_channels=[256, 256])
        # 64 -> 256 : interpolated features gives 256 + skip has 128 = 384
        self.fp3 = PointNetFeaturePropagation(in_channels=256 + 128, mlp_channels=[256, 256])
        # 256 -> 1024 : interpolated features gives 256 + skip has 64 = 320
        self.fp2 = PointNetFeaturePropagation(in_channels=256 + 64, mlp_channels=[256, 128])
        # 1024 -> N : interpolated features gives 128 + skip has D (if any) = 128 + extra_channels
        self.fp1 = PointNetFeaturePropagation(in_channels=128 + extra_channels, mlp_channels=[128, 128])

        # -----------------------
        # Per-point classifier head
        # -----------------------
        self.classifier = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Conv1d(128, num_classes, kernel_size=1)
        )

    def forward(self, points):
        """
        Input:
            points:    [B, N, C]  (xyz + optional features)
        Ouput:
            log_probs: [B, num_classes, N]
        """
        assert points.ndim == 3, f"Expected [B,N,C] tensor, got shape {points.shape}"
        assert points.shape[2] == 3 + self.extra_channels, \
            f"Expected {3 + self.extra_channels} channels (xyz + {self.extra_channels}), got {points.shape[2]}"

        xyz0 = points[:, :, :3]

        if points.shape[2] > 3:
            feat0 = points[:, :, 3:]
        else:
            feat0 = None

        # -------- Encoder --------
        xyz1, feat1 = self.sa1(xyz0, feat0)   # xyz1 [B,1024,3], feat1 [B,1024,64]
        xyz2, feat2 = self.sa2(xyz1, feat1)   # xyz2 [B,256,3],  feat2 [B,256,128]
        xyz3, feat3 = self.sa3(xyz2, feat2)   # xyz3 [B,64,3],   feat3 [B,64,256]
        xyz4, feat4 = self.sa4(xyz3, feat3)   # xyz4 [B,16,3],   feat4 [B,16,512]

        # -------- Decoder --------
        # FP4: upsample from 16 -> 64
        feat3_up = self.fp4(target_xyz=xyz3,
                            source_xyz=xyz4,
                            source_features=feat4,
                            target_skip_features=feat3)  # [B,64,256]
        # FP3: upsample from 64 -> 256
        feat2_up = self.fp3(target_xyz=xyz2,
                            source_xyz=xyz3,
                            source_features=feat3_up,
                            target_skip_features=feat2)  # [B,256,256]
        # FP2: upsample from 256 -> 1024
        feat1_up = self.fp2(target_xyz=xyz1,
                            source_xyz=xyz2,
                            source_features=feat2_up,
                            target_skip_features=feat1)  # [B,1024,128]
        # FP1: upsample from 1024 -> N
        feat0_up = self.fp1(target_xyz=xyz0,
                            source_xyz=xyz1,
                            source_features=feat1_up,
                            target_skip_features=feat0)  # [B,N,128]

        # -------- Head --------
        x = feat0_up.permute(0, 2, 1).contiguous()      # [B,128,N]
        logits = self.classifier(x)                     # [B,num_classes,N]
        log_probs = F.log_softmax(logits, dim=1)        # [B,num_classes,N]
        
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

    network = PointNetPlusPlusClassifier(num_classes=10, extra_channels=0, dropout=0.5).to(device)
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