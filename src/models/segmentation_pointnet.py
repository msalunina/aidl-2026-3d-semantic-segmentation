"""
https://colab.research.google.com/drive/14mmIaefEVs6Ro6_QoeBBFfr1Zpv7RD9M?authuser=2#scrollTo=ycw_6xYaHiyf
https://arxiv.org/pdf/1612.00593
https://datascienceub.medium.com/pointnet-implementation-explained-visually-c7e300139698
https://miro.medium.com/v2/format:webp/1*jzi1PoxGk5r9t1q1Q_zdfA.png
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_pointnet import BasePointNet

class SegmentationPointNet(nn.Module):

    def __init__(self, point_dimension=3, global_feature_size=1024, num_classes=4, dropout=0.3):
        super(SegmentationPointNet, self).__init__()
    
        #add one hot vector for class ?¿
        self._feature_size = global_feature_size
        self._num_classes = num_classes
        self.base_pointnet = BasePointNet(point_dimension=point_dimension, global_feat_size=global_feature_size)

        self.conv_1 = nn.Conv1d(64+global_feature_size, 512, 1)
        self.conv_2 = nn.Conv1d(512, 256, 1)
        self.conv_3 = nn.Conv1d(256, 128, 1)

        self.conv_4 = nn.Conv1d(128, 128, 1)
        self.conv_5 = nn.Conv1d(128, num_classes, 1)

        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)
        self.bn_3 = nn.BatchNorm1d(128)
        self.bn_4 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        
        bsz = x.shape[0]
        num_points = x.shape[1]
        
        global_feature_vector, point_features, feature_transform, tnet_out, ix_maxpool = self.base_pointnet(x)
        
        #concatenar las entradas
        point_features_expanded = global_feature_vector.repeat(1, 1, num_points)
        point_features_expanded = global_feature_vector.reshape(bsz, point_features.shape[1], num_points)
        
        x = torch.cat([feature_transform, point_features_expanded], dim=1)

        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        
        point_features = x
        
        x = self.dropout(x)
        x = self.conv_5(x)
        x = F.log_softmax(x, dim=1)
        
        return  x
