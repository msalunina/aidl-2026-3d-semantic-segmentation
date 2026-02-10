"""
https://colab.research.google.com/drive/14mmIaefEVs6Ro6_QoeBBFfr1Zpv7RD9M?authuser=2#scrollTo=ycw_6xYaHiyf
https://arxiv.org/pdf/1612.00593
https://datascienceub.medium.com/pointnet-implementation-explained-visually-c7e300139698
https://miro.medium.com/v2/format:webp/1*jzi1PoxGk5r9t1q1Q_zdfA.png
"""

import torch.nn as nn
import torch.nn.functional as F
from .base_pointnet import BasePointNet

class ClassificationPointNet(nn.Module):

    def __init__(self, point_dimension=3, global_feat_size=1024, num_classes=4, dropout=0.3):
        super(ClassificationPointNet, self).__init__()
        
        self._global_feat_size = global_feat_size
        self._num_classes = num_classes
        
        self.base_pointnet = BasePointNet(point_dimension=point_dimension, global_feat_size=global_feat_size)

        self.fc_1 = nn.Linear(global_feat_size, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, num_classes)

        self.bn_1 = nn.LayerNorm(512)
        self.bn_2 = nn.LayerNorm(256)

        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x):
        x, point_features, feature_transform, tnet_out, ix_maxpool = self.base_pointnet(x)
        
        x = F.relu(self.bn_1(self.fc_1(x)))
        x = F.relu(self.bn_2(self.fc_2(x)))
        x = self.dropout_1(x)
        x = self.fc_3(x)
        x = F.log_softmax(x, dim=1)
        
        return x, tnet_out, ix_maxpool
    