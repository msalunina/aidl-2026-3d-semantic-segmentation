"""
PointNet for 3D Classification and Segmentation.
Based on the paper: PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
 

class TransformationNet(nn.Module):
    """
    T-Net: Transformation Network for learning spatial transformations
    Used for input transform and feature transform
    """

    def __init__(self, input_dim, output_dim):
        super(TransformationNet, self).__init__()
        self.output_dim = output_dim

        # Shared MLP layers 
        self.conv_1 = nn.Conv1d(input_dim, 64, 1)       # weight matrix: [64,input_dim] (W * x = x')
        self.conv_2 = nn.Conv1d(64, 128, 1)             # weight matrix: [128,64]
        self.conv_3 = nn.Conv1d(128, 1024, 1)           # weight matrix: [1024,128]

        # Batch normalization
        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(1024)
        self.bn_4 = nn.BatchNorm1d(512)
        self.bn_5 = nn.BatchNorm1d(256)

        # Fully connected layers
        self.fc_1 = nn.Linear(1024, 512)                # weight matrix: [512,1024]
        self.fc_2 = nn.Linear(512, 256)                 # weight matrix: [256,512]
        self.fc_3 = nn.Linear(256, self.output_dim * self.output_dim)   # weight matrix: [output_dim x output_dim, 256]

        # OLGA: I added this on mine:
        # Weird fix to initialize last layers params to 0 and force an identity matrix as start.
        # The initial value for the 3x3 (64x64) is x+identity, but x is not 0 by default so it may add something
        # even if small. Forcing params to 0, guarantees that it will start with identity.
        # nn.init.zeros_(self.fc_3.weight)
        # nn.init.zeros_(self.fc_3.bias)

    def forward(self, x):
        # x: [B, N, C]

        # nn.Conv1 expects [B, C, N])
        num_points = x.shape[1]
        x = x.transpose(2, 1)                           # [B, input_dim, N]

        # Shared MLP
        x = F.relu(self.bn_1(self.conv_1(x)))           # [B, 64, N]
        x = F.relu(self.bn_2(self.conv_2(x)))           # [B, 128, N]
        x = F.relu(self.bn_3(self.conv_3(x)))           # [B, 1024, N]

        # Max pooling
        x = nn.MaxPool1d(num_points)(x)                 # [B, 1024, 1]
        # TODO: update to accomodate batch size 1
        # Actual problem is BatchNorm, it cant estimate variance with 1 sample!!!!!
        x = x.view(-1, 1024)                            # [B, 1024] (removes dimensions of 1)

        # Fully connected layers
        x = F.relu(self.bn_4(self.fc_1(x)))             # [B, 512]
        x = F.relu(self.bn_5(self.fc_2(x)))             # [B, 256]
        x = self.fc_3(x)                                # [B, output_dim x output_dim]

        # identity_matrix = torch.eye(self.output_dim)    # [output_dim, output_dim]
        # if torch.cuda.is_available():
        #     identity_matrix = identity_matrix.cuda()
        # OLGA: instead of the 3 lines above,
        # create identity matrix directly on the device of x
        identity_matrix = torch.eye(self.output_dim, device=x.device).unsqueeze(0)  # [1, output_dim, output_dim]         
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix          # [B, output_dim, output_dim]
        return x
    

class PointNetBackbone(nn.Module):
    """
    PointNet feature extraction backbone
    """

    # TODO: add concatenation of additional channels if input_channels > 3
    # after input transformation! DONE!

    def __init__(self, input_channels=3):
        super(PointNetBackbone, self).__init__()
        self.input_channels = input_channels

        # Input transformation
        # self.input_tnet = TransformationNet(input_dim=input_channels, output_dim=input_channels) 
        
        # Input transform with additional channels:
        # Force input_dim=3 and output_dim=3 (x,y,z)
        self.input_tnet = TransformationNet(input_dim=3, output_dim=3) 

        # Shared MLP(64, 64)
        self.conv_1 = nn.Conv1d(input_channels, 64, 1)      # weight matrix: [64,C+3]
        self.conv_2 = nn.Conv1d(64, 64, 1)                  # weight matrix: [64,64]
        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        
        # Feature transformation
        self.feature_tnet = TransformationNet(input_dim=64, output_dim=64)       
        
        # Shared MLP(64, 128, 1024)
        self.conv_3 = nn.Conv1d(64, 64, 1)                  # weight matrix: [64,64]
        self.conv_4 = nn.Conv1d(64, 128, 1)                 # weight matrix: [128,64]
        self.conv_5 = nn.Conv1d(128, 1024, 1)               # weight matrix: [1024,128]
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(1024)
        
    def forward(self, x): 
        # x: [B, N, 3] 
        # Input transformation
        # input_transform = self.input_tnet(x)                # [B, 3, 3] T-net tensor
        # x = torch.bmm(x, input_transform)                   # [B, N, 3] Batch matrix-matrix product
        # x = x.transpose(2, 1)                               # [B, 3, N] transpose for Conv1d

        # x: [B, N, 3+C] 
        # Input transform with additional channels:
        xyz = x[:, :, :3]                                   # [B, N, 3]
        input_transform = self.input_tnet(xyz)              # [B, 3, 3] T-net tensor
        xyz = torch.bmm(xyz, input_transform)               # [B, N, 3] Batch matrix-matrix product

        if x.shape[2]>3:
            extra_channels = x[:, :, 3:]                    # [B, N, C]
            x = torch.cat([xyz, extra_channels], dim=2)     # [B, N, 3+C]
        else:  
            x = xyz
        x = x.transpose(2, 1)                               # [B, 3+C, N] transpose for Conv1d

        # Shared MLP(64, 64)
        x = F.relu(self.bn_1(self.conv_1(x)))               # [B, 64, N]
        x = F.relu(self.bn_2(self.conv_2(x)))               # [B, 64, N]

        # Feature transformation
        x = x.transpose(2, 1)                               # [B, N, 64] transpose for feature transform
        feature_transform = self.feature_tnet(x)            # [B, 64, 64] T-Net tensor 
        x = torch.bmm(x, feature_transform)                 # [B, N, 64] local point features
        x = x.transpose(2, 1)                               # [B, 64, N] transpose back for Conv1d

        # Save point features for segmentation
        point_features = x.clone()                          # [B, 64, N]

        # Shared MLP(64, 128, 1024)
        x = F.relu(self.bn_3(self.conv_3(x)))               # [B, 64, N]
        x = F.relu(self.bn_4(self.conv_4(x)))               # [B, 128, N]
        x = F.relu(self.bn_5(self.conv_5(x)))               # [B, 1024, N]

        # Global feature (max pooling)
        global_feature = torch.max(x, 2, keepdim=True)[0]   # [B, 1024, 1] because of keepdim=True
                
        return feature_transform, point_features, global_feature
    

class PointNetClassification(nn.Module):
    """
    PointNet for 3D Classification
    """

    def __init__(self, num_classes, input_channels=3, dropout=0.3):
        super(PointNetClassification, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout = nn.Dropout(p=dropout)

        # Feature extraction backbone
        self.backbone = PointNetBackbone(
            input_channels=input_channels
        )

        # Classification head: MLP(512, 256, num_classes)
        # Fully connected linear layers
        self.fc_1 = nn.Linear(1024, 512)                    # weight matrix: [512,1024] 
        self.fc_2 = nn.Linear(512, 256)                     # weight matrix: [256,512] 
        self.fc_3 = nn.Linear(256, num_classes)             # weight matrix: [num_classes,256] 
        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)


    def forward(self, x):
        # x: [B, N, 3] or [B, N, 3+C]
        
        feature_transform, point_features, global_feature = self.backbone(x)
        # global feature: [B, 1024, 1], but nn.Linear expects [B, 1024]

        global_feature = global_feature.view(-1, 1024)      # [B, 1024] removes the last 1 dimension 
        x = F.relu(self.bn_1(self.fc_1(global_feature)))    # [B, 512]
        x = F.relu(self.bn_2(self.fc_2(x)))                 # [B, 256]
        x = self.dropout(x)
        x = F.log_softmax(self.fc_3(x), dim=1)              # [B, num_classes}]

        return feature_transform, x
    

class PointNetSegmentation(nn.Module):
    """
    PointNet for 3D Semantic Segmentation
    """

    def __init__(self, num_classes, input_channels=3, dropout=0.3):
        super(PointNetSegmentation, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        # OLGA: paper says there is no dropout for segmentation
        # self.dropout = nn.Dropout(p=dropout)    

        # Feature extraction backbone
        self.backbone = PointNetBackbone(
            input_channels=input_channels
        )

        # Segmentation head: 
        # MLP(512, 256, 128)
        self.conv_1 = nn.Conv1d(1088, 512, 1)                   # weight matrix: [512,64+1024]
        self.conv_2 = nn.Conv1d(512, 256, 1)                    # weight matrix: [256,512]
        self.conv_3 = nn.Conv1d(256, 128, 1)                    # weight matrix: [128,256]
        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)
        self.bn_3 = nn.BatchNorm1d(128)

        # MLP(128,num_classes)
        # OLGA: according to Fig 2 in the paper, the Segmentation head has
        # one extra 128 layer (self.conv_4 + self.bn_4) before num_classes
        self.conv_4 = nn.Conv1d(128, 128, 1)                    # weight matrix: [128,128]         
        self.conv_5 = nn.Conv1d(128, num_classes, 1)            # weight matrix: [num_classes,128]
        self.bn_4 = nn.BatchNorm1d(128)

    def forward(self, x):
        # x: [B, N, 3] or [B, N, 3+C]

        feature_transform, point_features, global_feature = self.backbone(x)
        # point_features: [B, 64, N]
        # global feature: [B, 1024, 1]
        
        num_points = x.shape[1]
        global_feature_expanded = global_feature.repeat(1, 1, num_points)   # [B, 1024, N]

        # Concatenate point features and global feature
        x = torch.cat([point_features, global_feature_expanded], dim=1)     # [B, 64+1024, N]

        x = F.relu(self.bn_1(self.conv_1(x)))               # [batch, 512, nPoints]
        x = F.relu(self.bn_2(self.conv_2(x)))               # [batch, 256, nPoints]
        x = F.relu(self.bn_3(self.conv_3(x)))               # [batch, 128, nPoints]
        # This line can be commented if we simplify the extra 128 layer.
        x = F.relu(self.bn_4(self.conv_4(x)))               # [batch, 128, nPoints] 
        x = F.log_softmax(self.conv_5(x), dim=1)            # [batch, num_classes, nPoints]

        return feature_transform, x

