# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# INCIS:
# C = input featues (channels)
# F = output features (filters)
# N = number of points
# B = batch_size
#
#   LAYER              INPUT SHAPE   WIGHT SHAPE           BIAS SHAPE    OUTPUT SHAPE
# nn.Linear(C → F)	     [B, C]	       [F, C]	              [F]	       [B, F]
# nn.Conv1d(C → F, k=1)	 [B, C, N]	   [F, C, 1] (≡ [F, C])	  [F]	       [B, F, N]


class TransformationNet(nn.Module):
    # For each pointset (object), predicts a transformation matrix (3x3 or 64x64) to make objects "canonical"
    # INPUT:  [batch, nPoints, 3]  -->  OUTPUT: [batch, 3, 3]      (channel=3)
    # INPUT:  [batch, nPoints, 64] -->  OUTPUT: [batch, 64, 64]    (channel=64)

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim

        # nn.Conv1d expects [batch,channel,nPoints]=[B,C,N] so in the forward, we will need to transpose last 2 dimensions.
        # Ignoring batch dimension:_
        # For C=3, after transposing, the input for nn.Conv1d will be [3,nPoints]:
        #  W * x = x'  --> [64,3] * [3,nPoints] = [64,nPoints]
        self.conv_1 = nn.Conv1d(input_dim, 64, 1)                   # weight matrix: [64,3]     --> output: [64,nPoints] 
        self.conv_2 = nn.Conv1d(64, 128, 1)                         # weight matrix: [128,64]   --> output: [128,nPoints] 
        self.conv_3 = nn.Conv1d(128, 1024, 1)                       # weight matrix: [1024,128] --> output: [1024,nPoints] 
        # Normalizes along features considering all objects and all points in the batch.
        # For instance: take feature #39 (out of 64, in this case) of each point and each opbject and normalize them.
        # After normalizing, the collection of features #39 will have mean=0 and std=1
        self.bn_1 = nn.BatchNorm1d(64)  
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(1024)

        # Fully connected linear layers after maxPool (unlike nn.Conv1d, there is no nPoints dimension here)
        self.fc_1 = nn.Linear(1024, 512)                            # weight matrix: [512,1024] --> output: [512] 
        self.fc_2 = nn.Linear(512, 256)                             # weight matrix: [256,512]  --> output: [256] 
        self.fc_3 = nn.Linear(256, self.output_dim*self.output_dim) # weight matrix: [3x3,256]  --> output: [3x3] 
        self.bn_4 = nn.BatchNorm1d(512)
        self.bn_5 = nn.BatchNorm1d(256)

        # Weird fix to initialize last layers params to 0 and force an identity matrix as start.
        # The initial value for the 3x3 is x+identity, but x is not 0 by default so it may add something
        # even if small. Forcing params to 0, guarantees that it will start with identity.
        nn.init.zeros_(self.fc_3.weight)
        nn.init.zeros_(self.fc_3.bias)

    def forward(self, x):
        num_points = x.shape[1]
        # My input shape is [batch, nPoints, coordinates]. nn.Conv1d wants [batch, coordinates, nPoints]. 
        # In fact it is not pointNet who wants that but Pythorch by how nn.Conv1d are defined: 
        # [batch, channels=coordinates, nPoints] (CHANNELS FIRST)
        # So, we first need to transpose last 2 dimensions.
        x = x.transpose(2, 1)   
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        # we have: [1024,N] and we want the maximum value of each feature: --> [1024,1]
        # For each feature, we choose the feature value of the point that maximises it 
        # The output is a GLOBAL FEATURE VECTOR DESCRIBING THE OBJECT [1024,1]
        # NOTA: we can't define maxPool1d in __init__ because num_points is only known at runtime.
        x = nn.MaxPool1d(num_points)(x)         # nn.MaxPool1d(kernel_size=num_points)
        x = x.view(-1, 1024)                     # "removes" the 1 dimension

        x = F.relu(self.bn_4(self.fc_1(x)))     # from [1024] to [512]: mixing
        x = F.relu(self.bn_5(self.fc_2(x)))     # from [512] to [256]: compress
        x = self.fc_3(x)                        # from [256] to [3x3=9] or [64x64=4096]

        # Reshape the vector into a matrix.
        # Add identity so that the transformation is only adding variatons.
        # If variations were close to 0 and we wouldn't have identity, we would "lose" everything.
        identity_matrix = torch.eye(self.output_dim, device=x.device)
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        return x


class BasePointNet(nn.Module):
    # Two different kinds of matrices:
    # - Tnet (e.g input_transform 3x3): they are dynamic and computed per object (they depen on the point cloud, x)
    # - Weight matrices (e.g. conv1d weights): learned, they belong to the model

    def __init__(self, point_dimension):
        super().__init__()
        self.input_tnet = TransformationNet(input_dim=point_dimension, output_dim=point_dimension)
        self.feature_tnet = TransformationNet(input_dim=64, output_dim=64)
        # self.input_tnet_function = self.input_tnet
        # self.feature_tnet_function = self.feature_tnet

        # Shared MLP after input_transform (64,64)
        self.conv_1 = nn.Conv1d(point_dimension, 64, 1) # weight matrix: [64,3]    --> output: [64,nPoints] 
        self.conv_2 = nn.Conv1d(64, 64, 1)              # weight matrix: [64,64]   --> output: [64,nPoints] 
        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)

        # Shared MLP after feature_transform (64,128,1024)  
        self.conv_3 = nn.Conv1d(64, 64, 1)              # weight matrix: [64,64]    --> output: [64,nPoints]       
        self.conv_4 = nn.Conv1d(64, 128, 1)             # weight matrix: [128,64]   --> output: [128,nPoints] 
        self.conv_5 = nn.Conv1d(128, 1024, 1)           # weight matrix: [1024,128] --> output: [1024,nPoints] 
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(1024)


    def forward(self, x, plot=False):
        num_points = x.shape[1]

        # INPUT x: [batch, nPoints, 3]
        # self.input_transform is a "function", it depends on x: 
        # - we pass x, we compute its matrix and then multiply (perform the transformation)                             
        input_tnet = self.input_tnet(x)        # T-Net tensor:                 [batch, 3, 3]
        x = torch.bmm(x, input_tnet)           # Batch matrix-matrix product   [batch, nPoints, 3] POINTS TRANSFORMED

        x = x.transpose(2, 1)                                   # bcause nn.Conv1d needs input: [batch, 3, nPoints]                
        x = F.relu(self.bn_1(self.conv_1(x)))                   #                               [batch, 64, nPoints]
        x = F.relu(self.bn_2(self.conv_2(x)))                   #                               [batch, 64, nPoints]
        x = x.transpose(2, 1)                                   # because Tnet needs input:     [batch, nPoints, 64]       
   
        feature_tnet= self.feature_tnet(x)    # T-Net tensor                  [batch, 64, 64]
        x = torch.bmm(x, feature_tnet)              # Batch matrix-matrix product   [batch, nPoints, 64]
        point_features = x

        x = x.transpose(2, 1)                                   # bcause nn.Conv1d needs input: [batch, 64, nPoints] 
        x = F.relu(self.bn_3(self.conv_3(x)))                   #                               [batch, 64, nPoints] 
        x = F.relu(self.bn_4(self.conv_4(x)))                   #                               [batch, 128, nPoints] 
        x = F.relu(self.bn_5(self.conv_5(x)))                   #                               [batch, 1024, nPoints] 
        # max-pooling. It always acts on last dimension, which is nPoints: 
        # for each feature, which of the points shows a highest value "take the max over all points"
        x, ix = nn.MaxPool1d(kernel_size=num_points, return_indices=True)(x)
        global_features = x.view(-1, 1024)                      # global feature vector         [batch, 1024]

        # global_features     --> global feature vector describing the object
        # ix                  --> nice for checking which point index contributed to the global feature
        # point_features      --> needed for segmentation
        # input_tnet_tensor   --> matrix used to transform points (canonical) transfromed input points nice for visualization
        # feature_tnet_tensor --> matrix needed for loss (regularization)
        return global_features, ix, point_features, feature_tnet, input_tnet


    

class ClassificationPointNet(nn.Module):

    def __init__(self, num_classes, dropout=0.3, point_dimension=3):
        super().__init__()
        self.base_pointnet = BasePointNet(point_dimension=point_dimension)

        # Fully connected linear layers (512,256,num_classes)
        self.fc_1 = nn.Linear(1024, 512)                 # weight matrix: [512,1024]        --> output: [512] 
        self.fc_2 = nn.Linear(512, 256)                  # weight matrix: [256,512]         --> output: [256] 
        self.fc_3 = nn.Linear(256, num_classes)          # weight matrix: [num_classes,256] --> output: [num_classes] 
        
        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)
        
        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x):
        global_features, ix_maxpool, point_features, feature_tnet, input_tnet = self.base_pointnet(x)

        x = F.relu(self.bn_1(self.fc_1(global_features)))
        x = F.relu(self.bn_2(self.fc_2(x)))
        x = self.dropout_1(x)
        log_probs = F.log_softmax(self.fc_3(x), dim=1)

        return log_probs, ix_maxpool, point_features, feature_tnet, input_tnet



class SegmentationPointNet(nn.Module):
    def __init__(self, num_classes, dropout=0.3, point_dimension=3):
        super().__init__()
        self.base_pointnet = BasePointNet(point_dimension=point_dimension)

        # Shared MLP (512,256,128)
        self.conv_1 = nn.Conv1d(64+1024, 512, 1)          # weight matrix: [512,64+1024] --> output: [512,nPoints] 
        self.conv_2 = nn.Conv1d(512, 256, 1)              # weight matrix: [256,512]     --> output: [256,nPoints] 
        self.conv_3 = nn.Conv1d(256, 128, 1)              # weight matrix: [128,256]     --> output: [128,nPoints] 
        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)
        self.bn_3 = nn.BatchNorm1d(128)

        self.conv_4 = nn.Conv1d(128, 128, 1)              # weight matrix: [128,128]         --> output: [128,nPoints] 
        self.conv_5 = nn.Conv1d(128, num_classes, 1)      # weight matrix: [num_classes,128] --> output: [num_classes,nPoints] 
        self.bn_4 = nn.BatchNorm1d(128)

        self.dropout_1 = nn.Dropout(dropout)


    def forward(self, x):
        global_features, ix_maxpool, point_features, feature_tnet, input_tnet = self.base_pointnet(x)
        # [B,1024] global features vector
        # [B,N,64] point features vector
        nPoints = point_features.shape[1]
        global_features = global_features.unsqueeze(1)              # [B,1,1024]
        global_features = global_features.expand(-1, nPoints , -1)  # [B,N,1024]
        x =  torch.cat([point_features, global_features], dim=2)

        x = x.transpose(2, 1)                                       # [B, 1024+64, N]
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        x = F.relu(self.bn_4(self.conv_4(x)))
        x = self.dropout_1(x)
        log_probs = F.log_softmax(self.conv_5(x), dim=1)

        return log_probs, ix_maxpool, point_features, feature_tnet, input_tnet

