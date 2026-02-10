"""
https://colab.research.google.com/drive/14mmIaefEVs6Ro6_QoeBBFfr1Zpv7RD9M?authuser=2#scrollTo=ycw_6xYaHiyf
https://arxiv.org/pdf/1612.00593
https://datascienceub.medium.com/pointnet-implementation-explained-visually-c7e300139698
https://miro.medium.com/v2/format:webp/1*jzi1PoxGk5r9t1q1Q_zdfA.png
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tnet import TransformationNet

class BasePointNet(nn.Module):

    def __init__(self, point_dimension, global_feat_size):
        super(BasePointNet, self).__init__()
        self.input_transform = TransformationNet(input_dim=point_dimension, output_dim=point_dimension)        
        self.feature_transform = TransformationNet(input_dim=64, output_dim=64)
        
        self._global_feat_size = global_feat_size

        self.conv_1 = nn.Conv1d(point_dimension, 64, 1)
        self.conv_2 = nn.Conv1d(64, 64, 1)
        
        self.conv_3 = nn.Conv1d(64, 64, 1)
        self.conv_4 = nn.Conv1d(64, 128, 1)
        self.conv_5 = nn.Conv1d(128, global_feat_size, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(global_feat_size)


    def forward(self, x):
        num_points = x.shape[1]

        input_transform = self.input_transform(x) # T-Net tensor [batch, 3, 3]
        
        x = torch.bmm(x, input_transform) # Batch matrix-matrix product
        
        x = x.transpose(2, 1) #[B,n,3] -> [B,3,n]
        
        tnet_out=x.cpu().detach().numpy()
    
        x = F.relu(self.bn_1(self.conv_1(x))) #[B, 64, n]
        x = F.relu(self.bn_2(self.conv_2(x))) #[B, 64, n]
        
        x = x.transpose(2, 1) #[B,64,n] -> [B,n,64]

        feature_transform = self.feature_transform(x) # T-Net tensor [batch, 64, 64]
        
        x = torch.bmm(x, feature_transform)
        x = x.transpose(2, 1) #[B, n, 64]
        point_features = x #copy the result of the last feature transform, [B, n, 64]
        
        x = F.relu(self.bn_3(self.conv_3(x))) #[B, n, 64]
        x = F.relu(self.bn_4(self.conv_4(x))) #[B, n, 128]
        x = F.relu(self.bn_5(self.conv_5(x))) #[B, n, 1024]
        
        x, ix = nn.MaxPool1d(num_points, return_indices=True)(x)  # max-pooling
        
        x = x.view(-1, self._global_feat_size)  # global feature vector

        return x, point_features, feature_transform, tnet_out, ix