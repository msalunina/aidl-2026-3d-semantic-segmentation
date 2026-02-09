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
        self.conv_1 = nn.Conv1d(input_dim, 64, 1)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 1024, 1)

        # Batch normalization
        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(1024)
        self.bn_4 = nn.BatchNorm1d(512)
        self.bn_5 = nn.BatchNorm1d(256)

        # Fully connected layers
        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, self.output_dim * self.output_dim)

    def forward(self, x):
        num_points = x.shape[1]
        x = x.transpose(2, 1)

        # Shared MLP
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        # Max pooling
        x = nn.MaxPool1d(num_points)(x)
        x = x.view(-1, 1024) # TODO: update to accomodate batch size 1

        # Fully connected layers
        x = F.relu(self.bn_4(self.fc_1(x)))
        x = F.relu(self.bn_5(self.fc_2(x)))
        x = self.fc_3(x)

        identity_matrix = torch.eye(self.output_dim)
        if torch.cuda.is_available():
            identity_matrix = identity_matrix.cuda()
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        return x
    

class PointNetBackbone(nn.Module):
    """
    PointNet feature extraction backbone
    """

    # TODO: add concatenation of additional channels if input_channels > 3
    # after input transformation

    def __init__(self, input_channels=3):
        super(PointNetBackbone, self).__init__()
        self.input_channels = input_channels

        # Input transformation
        self.input_tnet = TransformationNet(input_channels, input_channels)

        # MLP(64, 64)
        self.conv_1 = nn.Conv1d(input_channels, 64, 1)
        self.conv_2 = nn.Conv1d(64, 64, 1)
        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        
        # Feature transformation
        self.feature_tnet = TransformationNet(64, 64)
        
        # MLP(64, 128, 1024)
        self.conv_3 = nn.Conv1d(64, 64, 1)
        self.conv_4 = nn.Conv1d(64, 128, 1)
        self.conv_5 = nn.Conv1d(128, 1024, 1)
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(1024)
        
    def forward(self, x):        
        # Input transformation
        input_transform = self.input_tnet(x) # T-Net tensor [batch, 3, 3]
        x = torch.bmm(x, input_transform) # Batch matrix-matrix product [batch, num_points, 3]
        x = x.transpose(2, 1) # Transpose to [batch, 3, num_points] for Conv1d
        
        # MLP(64, 64)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))

        # Feature transformation
        x = x.transpose(2, 1) # Transpose to [batch, num_points, 64] for feature transform
        feature_transform = self.feature_tnet(x)  # T-Net tensor [batch, 64, 64]
        x = torch.bmm(x, feature_transform)  # local point features [batch, num_points, 64]
        x = x.transpose(2, 1) # Transpose back to [batch, 64, num_points] for Conv1d

        # Save point features for segmentation
        point_features = x.clone()

        # MLP(64, 128, 1024)
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = F.relu(self.bn_5(self.conv_5(x)))

        # Global feature (max pooling)
        global_feature = torch.max(x, 2, keepdim=True)[0]

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
        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, num_classes)

        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)


    def forward(self, x):
        feature_transform, point_features, global_feature = self.backbone(x)

        x = F.relu(self.bn_1(self.fc_1(global_feature.view(-1, 1024))))
        x = F.relu(self.bn_2(self.fc_2(x)))
        x = self.dropout(x)
        x = F.log_softmax(self.fc_3(x), dim=1)

        return feature_transform, x
    

class PointNetSegmentation(nn.Module):
    """
    PointNet for 3D Semantic Segmentation
    """

    def __init__(self, num_classes, input_channels=3, dropout=0.3):
        super(PointNetSegmentation, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout = nn.Dropout(p=dropout)

        # Feature extraction backbone
        self.backbone = PointNetBackbone(
            input_channels=input_channels
        )

        # Segmentation head: MLP(512, 256, 128, num_classes)
        self.conv_1 = nn.Conv1d(1088, 512, 1)
        self.conv_2 = nn.Conv1d(512, 256, 1)
        self.conv_3 = nn.Conv1d(256, 128, 1)
        self.conv_4 = nn.Conv1d(128, num_classes, 1)


    def forward(self, x):
        feature_transform, point_features, global_feature = self.backbone(x)

        num_points = x.shape[1]
        global_feature_expanded = global_feature.repeat(1, 1, num_points)

        # Concatenate point features and global feature
        x = torch.cat([point_features, global_feature_expanded], dim=1)

        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = self.dropout(x)
        x = F.log_softmax(self.conv_4(x), dim=1)

        return feature_transform, x

