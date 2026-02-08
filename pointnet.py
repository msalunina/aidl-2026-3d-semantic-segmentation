"""
https://colab.research.google.com/drive/14mmIaefEVs6Ro6_QoeBBFfr1Zpv7RD9M?authuser=2#scrollTo=ycw_6xYaHiyf
https://arxiv.org/pdf/1612.00593
https://datascienceub.medium.com/pointnet-implementation-explained-visually-c7e300139698
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tnet import TransformationNet

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
        self.conv_5 = nn.Conv1d(128, 256, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(256)


    def forward(self, x):
        num_points = x.shape[1]

        input_transform = self.input_transform(x) # T-Net tensor [batch, 3, 3]
        x = torch.bmm(x, input_transform) # Batch matrix-matrix product
        x = x.transpose(2, 1)
        tnet_out=x.cpu().detach().numpy()

        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = x.transpose(2, 1)

        feature_transform = self.feature_transform(x) # T-Net tensor [batch, 64, 64]
        x = torch.bmm(x, feature_transform)
        x = x.transpose(2, 1)
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = F.relu(self.bn_5(self.conv_5(x)))
        x, ix = nn.MaxPool1d(num_points, return_indices=True)(x)  # max-pooling
        x = x.view(-1, self._global_feat_size)  # global feature vector

        return x, feature_transform, tnet_out, ix

class SegmentationPointNet(nn.Module):

    def __init__(self, point_dimension, transform_size, feature_size, num_classes, dropout):
        super(SegmentationPointNet, self).__init__()
    
        #add one hot vector for class ?¿
        self.transform_size = transform_size
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.base_pointnet = BasePointNet(point_dimension=point_dimension, global_feat_size=feature_size)

        self.conv_1 = nn.Conv1d(transform_size+feature_size, 512, 1)
        self.conv_2 = nn.Conv1d(512, 256, 1)
        self.conv_3 = nn.Conv1d(256, 128, 1)

        self.conv_4 = nn.Conv1d(transform_size*128, 128, 1)
        self.conv_5 = nn.Conv1d(128, num_classes, 1)

        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)
        self.bn_3 = nn.BatchNorm1d(128)
        self.bn_4 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(p=dropout)

    
    def forward(self,x):
        
        x, feature_transform, tnet_out, ix_maxpool = self.base_pointnet(x)
        
        #concatenar las entradas
        x = torch.cat([feature_transform, x],dim=1)

        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        
        point_features = x
        
        x = self.dropout(x)
        x = F.log_softmax(self.conv_5(x), dim=1)
        
        return point_features, x

class ClassificationPointNet(nn.Module):

    def __init__(self, num_classes, dropout=0.3, point_dimension=3, global_feat_size=256, seg_classes=4):
        super(ClassificationPointNet, self).__init__()
        self._global_feat_size = global_feat_size
        self._seg_classes = seg_classes
        self.base_pointnet = BasePointNet(point_dimension=point_dimension, global_feat_size=global_feat_size)

        self.fc_1 = nn.Linear(256, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, num_classes)

        self.bn_1 = nn.BatchNorm1d(128)
        self.bn_2 = nn.BatchNorm1d(64)

        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x):
        x, feature_transform, tnet_out, ix_maxpool = self.base_pointnet(x)
        
    
        x = F.relu(self.bn_1(self.fc_1(x)))
        x = F.relu(self.bn_2(self.fc_2(x)))
        x = self.dropout_1(x)

        return F.log_softmax(self.fc_3(x), dim=1), feature_transform, tnet_out, ix_maxpool
    







         
         
         