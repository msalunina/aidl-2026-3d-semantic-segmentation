# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PointNetSimple(nn.Module):
    def __init__(self, num_classes, point_dimension=3):
        super().__init__()
        self.conv1 = nn.Conv1d(point_dimension, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.bn4 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):                 # x: [B, N, 3]
        x = x.transpose(2, 1)            # [B, 3, N]
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))  # [B,128, N]
        x = F.relu(self.bn3(self.conv3(x)))  # [B,256, N]
        x = torch.max(x, dim=2).values       # max over N -> [B,256]

        x = F.relu(self.bn4(self.fc1(x)))    # [B,128]
        x = self.drop(x)
        logits = self.fc2(x)                 # [B,num_classes]
        return logits

# %%
# class PointNetClassification(nn.Module):
    
class TransformationNet(nn.Module):
    # For each pointset (object), predicts a transformation matrix 3x3 to make them "canonical"
    # input: 3x3 matrix
    # output: 64x64
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim

        self.conv_1 = nn.Conv1d(input_dim, 64, 1)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 256, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(256)
        self.bn_4 = nn.BatchNorm1d(256)
        self.bn_5 = nn.BatchNorm1d(128)

        self.fc_1 = nn.Linear(256, 256)
        self.fc_2 = nn.Linear(256, 128)
        self.fc_3 = nn.Linear(128, self.output_dim*self.output_dim)

    def forward(self, x):
        num_points = x.shape[1]
        x = x.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        x = nn.MaxPool1d(num_points)(x)
        x = x.view(-1, 256)

        x = F.relu(self.bn_4(self.fc_1(x)))
        x = F.relu(self.bn_5(self.fc_2(x)))
        x = self.fc_3(x)

        identity_matrix = torch.eye(self.output_dim)
        if torch.cuda.is_available():
            identity_matrix = identity_matrix.cuda()
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        return x