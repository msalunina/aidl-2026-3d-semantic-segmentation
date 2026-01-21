"""
https://colab.research.google.com/drive/12RQDCV7krZtfjwJ0B4bOEBnvnDHTu-k2?usp=sharing
https://datascienceub.medium.com/pointnet-implementation-explained-visually-c7e300139698
https://arxiv.org/pdf/1612.00593
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformationNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(TransformationNet, self).__init__()
        self.output_dim = output_dim

        #column wise convolution [ k1 k2 k3] size 1, for each point in columns, outputs {64} features for each point
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
        #changes the input tensor from [B n 3] to [B 3 n]
        x = x.transpose(2, 1) 
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        x = nn.MaxPool1d(num_points)(x)
        x = x.view(-1, 256)

        x = F.relu(self.bn_4(self.fc_1(x)))
        x = F.relu(self.bn_5(self.fc_2(x)))
        #the size of the last linear layer is output x output, a vector
        x = self.fc_3(x)

        identity_matrix = torch.eye(self.output_dim)
        if torch.cuda.is_available():
            identity_matrix = identity_matrix.cuda()
        #reshapes x to have [B, output_dim, output_dim] - vector to matrix, then adds identity matrix
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        return x
