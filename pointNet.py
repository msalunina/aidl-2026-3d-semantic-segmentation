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


# %%    
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
        self.conv_1 = nn.Conv1d(input_dim, 64, 1)   # creates a weight matrix: [64,3]    --> output: [64,nPoints] 
        self.conv_2 = nn.Conv1d(64, 128, 1)         # creates a weight matrix: [128,64]  --> output: [128,nPoints] 
        self.conv_3 = nn.Conv1d(128, 256, 1)        # creates a weight matrix: [256,128] --> output: [256,nPoints] 

        # Normalizes along features considering all objects and all points in the batch.
        # For instance: take feature #39 (out of 64, in this case) of each point and each opbject and normalize them.
        # After normalizing, the collection of festures #39 will have mean=0 and std=1
        self.bn_1 = nn.BatchNorm1d(64)  
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(256)
        self.bn_4 = nn.BatchNorm1d(256)
        self.bn_5 = nn.BatchNorm1d(128)

        self.fc_1 = nn.Linear(256, 256)                             # weight matrix: [256,256] --> output: [256] 
        self.fc_2 = nn.Linear(256, 128)                             # weight matrix: [128,256] --> output: [128] 
        self.fc_3 = nn.Linear(128, self.output_dim*self.output_dim) # weight matrix: [3x3,128] --> output: [3x3] 

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

        # we have: [256,N] and we want the maximum value of each feature: --> [256,1]
        # For each feature, we choose the feature value of the point that maximises it 
        # The output is a GLOBAL FEATURE VECTOR DESCRIBING THE OBJECT [256,1]
        # NOTA: we can't define maxPool1d in __init__ because num_points is only known at runtime.
        x = nn.MaxPool1d(num_points)(x)         # nn.MaxPool1d(kernel_size=num_points)
        x = x.view(-1, 256)                     # "removes" the 1 dimension

        x = F.relu(self.bn_4(self.fc_1(x)))     # from [256] to [256]: mixing
        x = F.relu(self.bn_5(self.fc_2(x)))     # from [256] to [128]: compress
        x = self.fc_3(x)                        # from [128] to [3x3=9] or [64x64=4096]

        # Reshape the vector into a matrix.
        # Add identity so that the transformation is only adding variatons.
        # If variations were close to 0 and we wouldn't have identity, we would "lose" everything.
        identity_matrix = torch.eye(self.output_dim)
        if torch.cuda.is_available():
            identity_matrix = identity_matrix.cuda()
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        return x


class BasePointNet(nn.Module):
    # Two different kinds of matrices:
    # - Tnet (e.g input_transform 3x3): they are dynamic and computed per object (they depen on the point cloud, x)
    # - Weight matrices (e.g. conv1d weights): learned, they belong to the model

    def __init__(self, point_dimension):
        super().__init__()
        self.input_transform = TransformationNet(input_dim=point_dimension, output_dim=point_dimension)
        self.feature_transform = TransformationNet(input_dim=64, output_dim=64)

        self.conv_1 = nn.Conv1d(point_dimension, 64, 1) # weight matrix: [64,3]    --> output: [64,nPoints] 
        self.conv_2 = nn.Conv1d(64, 64, 1)              # weight matrix: [64,64]   --> output: [64,nPoints] 
        self.conv_3 = nn.Conv1d(64, 64, 1)              # weight matrix: [64,64]   --> output: [64,nPoints] 
        self.conv_4 = nn.Conv1d(64, 128, 1)             # weight matrix: [128,64]  --> output: [128,nPoints] 
        self.conv_5 = nn.Conv1d(128, 256, 1)            # weight matrix: [256,128] --> output: [256,nPoints] 

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(256)


    def forward(self, x, plot=False):
        num_points = x.shape[1]

        # INPUT x: [batch, nPoints, 3]
        # self.input_transform is a "function", it depends on x: 
        # - we pass x, we compute its matrix and then multiply (perform the transformation)                             
        input_transform = self.input_transform(x)           # T-Net tensor:                 [batch, 3, 3]
        x = torch.bmm(x, input_transform)                   # Batch matrix-matrix product   [batch, nPoints, 3]
        
        x = x.transpose(2, 1)                               # bcause nn.Conv1d needs input: [batch, 3, nPoints]                
        tnet_out=x.cpu().detach().numpy()
        x = F.relu(self.bn_1(self.conv_1(x)))               #                               [batch, 64, nPoints]
        x = F.relu(self.bn_2(self.conv_2(x)))               #                               [batch, 64, nPoints]
        x = x.transpose(2, 1)                               # because Tnet needs input:     [batch, nPoints, 64]       
   
        feature_transform = self.feature_transform(x)       # T-Net tensor                  [batch, 64, 64]
        x = torch.bmm(x, feature_transform)                 # Batch matrix-matrix product   [batch, nPoints, 64]
        
        x = x.transpose(2, 1)                               # bcause nn.Conv1d needs input: [batch, 64, nPoints] 
        x = F.relu(self.bn_3(self.conv_3(x)))               #                               [batch, 64, nPoints] 
        x = F.relu(self.bn_4(self.conv_4(x)))               #                               [batch, 128, nPoints] 
        x = F.relu(self.bn_5(self.conv_5(x)))               #                               [batch, 256, nPoints] 
        x, ix = nn.MaxPool1d(num_points, return_indices=True)(x)  # max-pooling
        x = x.view(-1, 256)                                 # global feature vector         [batch, 256]

        # x --> global feature vector describing the object
        # ix --> nice for checking which point index contributed to the global feature
        # tnet_out --> transfromed input points nice for visualization
        # feature_transform matrix --> needed for loss (regularization)
        return x, feature_transform, tnet_out, ix


    

class ClassificationPointNet(nn.Module):

    def __init__(self, num_classes, dropout=0.3, point_dimension=3):
        super().__init__()
        self.base_pointnet = BasePointNet(point_dimension=point_dimension)

        self.fc_1 = nn.Linear(256, 128)                 # weight matrix: [128,256]        --> output: [128] 
        self.fc_2 = nn.Linear(128, 64)                  # weight matrix: [64,128]         --> output: [64] 
        self.fc_3 = nn.Linear(64, num_classes)          # weight matrix: [num_classes,64] --> output: [num_classes] 

        self.bn_1 = nn.BatchNorm1d(128)
        self.bn_2 = nn.BatchNorm1d(64)

        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x):
        x, feature_transform, tnet_out, ix_maxpool = self.base_pointnet(x)

        x = F.relu(self.bn_1(self.fc_1(x)))
        x = F.relu(self.bn_2(self.fc_2(x)))
        x = self.dropout_1(x)
        logits = F.log_softmax(self.fc_3(x), dim=1)

        return logits#, feature_transform, tnet_out, ix_maxpool
