import torch
import torch.nn as nn
import torch.nn.functional as F

# INCIS:
# C = input featues (channels)
# F = output features (filters)
# N = number of points
# B = batch_size
#
#   LAYER              INPUT SHAPE   WIGHT SHAPE           BIAS SHAPE    OUTPUT SHAPE
# nn.Linear(C → F)	     [B, C]	       [F, C]	              [F]	       [B, F]
# nn.Conv1d(C → F, k=1)	 [B, C, N]	   [F, C, 1] (≡ [F, C])	  [F]	       [B, F, N]

# [B, C, N] means: “for each feature, how much each point exhibits it.”
# [B, N, C] means: “for each point, what features it has.”

# "PointNet.py" :
#  Tnet1:             3 --> (64,128,1024) + (512,256,3x3) --> 3 
#  Shared MLP:        3 --> (64,64)
#  Tnet2:            64 --> (64,128,1024) + (512,256,64x64) --> 64 
#  After Tnets:      64 --> (64,128,1024)
#  Classification: 1024 --> (512,246,classes)


class TransformationNet(nn.Module):
    # For each pointset (object), predicts a transformation matrix (3x3 or 64x64) to make objects "canonical"
    # INPUT:  [B, N, 3]  -->  OUTPUT: [B, 3, 3]      (channel=3)
    # INPUT:  [B, N, 64] -->  OUTPUT: [B, 64, 64]    (channel=64)

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim

        # nn.Conv1d expects [batch,channel,nPoints]=[B,C,N] so in the forward, we will need to transpose last 2 dimensions.
        # Ignoring batch dimension:_
        # For C=3, after transposing, the input for nn.Conv1d will be [3,nPoints]:
        #  W * x = x'  --> [64,3] * [3,nPoints] = [64,nPoints]
        self.conv_1 = nn.Conv1d(input_dim, 64, 1)                   # weight matrix: [64,3]     --> output: [B, 64, N] 
        self.conv_2 = nn.Conv1d(64, 128, 1)                         # weight matrix: [128,64]   --> output: [B, 128, N] 
        self.conv_3 = nn.Conv1d(128, 1024, 1)                       # weight matrix: [1024,128] --> output: [B, 1024, N] 
        # Normalizes along features considering all objects and all points in the batch.
        # For instance: take feature #39 (out of 64, in this case) of each point and each opbject and normalize them.
        # After normalizing, the collection of features #39 will have mean=0 and std=1
        self.bn_1 = nn.BatchNorm1d(64)  
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(1024)

        # Fully connected linear layers after maxPool (unlike nn.Conv1d, there is no nPoints dimension here)
        self.fc_1 = nn.Linear(1024, 512)                            # weight matrix: [512,1024] --> output: [B, 512] 
        self.fc_2 = nn.Linear(512, 256)                             # weight matrix: [256,512]  --> output: [B, 256] 
        self.fc_3 = nn.Linear(256, self.output_dim*self.output_dim) # weight matrix: [3x3,256]  --> output: [B, output_dim x output_dim] 
        self.bn_4 = nn.BatchNorm1d(512)
        self.bn_5 = nn.BatchNorm1d(256)

        # Weird fix to initialize last layers params to 0 and force an identity matrix as start.
        # The initial value for the 3x3 is x+identity, but x is not 0 by default so it may add something
        # even if small. Forcing params to 0, guarantees that it will start with identity.
        nn.init.zeros_(self.fc_3.weight)
        nn.init.zeros_(self.fc_3.bias)

    def forward(self, x):
        # x: [B, N, 3] / [B, N, 64]

        num_points = x.shape[1]
        # My input shape is [batch, nPoints, coordinates]. nn.Conv1d wants [batch, coordinates, nPoints]. 
        # In fact it is not pointNet who wants that but Pythorch by how nn.Conv1d are defined: 
        # [batch, channels=coordinates, nPoints] (CHANNELS FIRST)
        # So, we first need to transpose last 2 dimensions.
        x = x.transpose(2, 1)                               # [B, input_dim, N]   
        x = F.relu(self.bn_1(self.conv_1(x)))               # [B, 64, N]
        x = F.relu(self.bn_2(self.conv_2(x)))               # [B, 128, N]
        x = F.relu(self.bn_3(self.conv_3(x)))               # [B, 1024, N]

        # we have: [B, 1024, N] and we want the maximum value of each feature: --> [B, 1024, 1]
        # For each feature, we choose the feature value of the point that maximises it 
        # The output is a GLOBAL FEATURE VECTOR DESCRIBING THE OBJECT [B, 1024, 1]
        # NOTA: we can't define maxPool1d in __init__ because num_points is only known at runtime.
        x = nn.MaxPool1d(num_points)(x)                     # [B, 1024, 1] nn.MaxPool1d(kernel_size=num_points)
        x = x.view(-1, 1024)                                # [B, 1024] "reshapes": dim=1 with 1024, and dim=0 inferred with  
                                                            #            whatever is left to preserve number of elements
        x = F.relu(self.bn_4(self.fc_1(x)))                 # [B, 512]: mixing
        x = F.relu(self.bn_5(self.fc_2(x)))                 # [B, 256]: compress
        x = self.fc_3(x)                                    # [B, 3x3] or [B, 64x64]

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

        # Shared MLP after input_transform (64,64)
        self.conv_1 = nn.Conv1d(point_dimension, 64, 1) # weight matrix: [64,3]    --> output: [64, N] 
        self.conv_2 = nn.Conv1d(64, 64, 1)              # weight matrix: [64,64]   --> output: [64, N] 
        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)

        # Shared MLP after feature_transform (64,128,1024)  
        self.conv_3 = nn.Conv1d(64, 64, 1)              # weight matrix: [64,64]    --> output: [64, N]       
        self.conv_4 = nn.Conv1d(64, 128, 1)             # weight matrix: [128,64]   --> output: [128, N] 
        self.conv_5 = nn.Conv1d(128, 1024, 1)           # weight matrix: [1024,128] --> output: [1024, N] 
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(1024)


    def forward(self, x, plot=False):
        # x: [B, N, 3]

        num_points = x.shape[1]

        # self.input_transform is a "function", it depends on x: 
        # - we pass x, we compute its matrix and then multiply (perform the transformation)                             
        input_tnet = self.input_tnet(x)                         # T-Net tensor:                 [B, 3, 3]
        x = torch.bmm(x, input_tnet)                            # Batch matrix-matrix product   [B, N, 3] POINTS TRANSFORMED

        x = x.transpose(2, 1)                                   # bcause nn.Conv1d needs input: [B, 3, N]                
        x = F.relu(self.bn_1(self.conv_1(x)))                   #                               [B, 64, N]
        x = F.relu(self.bn_2(self.conv_2(x)))                   #                               [B, 64, N]
        x = x.transpose(2, 1)                                   # because Tnet needs input:     [B, N, 64]       
   
        feature_tnet= self.feature_tnet(x)                      # T-Net tensor                  [B, 64, 64]
        x = torch.bmm(x, feature_tnet)                          # Batch matrix-matrix product   [B, N, 64]
        point_features = x

        x = x.transpose(2, 1)                                   # bcause nn.Conv1d needs input: [B, 64, N] 
        x = F.relu(self.bn_3(self.conv_3(x)))                   #                               [B, 64, N] 
        x = F.relu(self.bn_4(self.conv_4(x)))                   #                               [B, 128, N] 
        x = F.relu(self.bn_5(self.conv_5(x)))                   #                               [B, 1024, N] 
        # max-pooling. It always acts on last dimension, which is nPoints: 
        # for each feature, which of the points shows a highest value "take the max over all points"
        x, ix = nn.MaxPool1d(kernel_size=num_points, return_indices=True)(x)
        global_features = x.view(-1, 1024)                      # global feature vector         [B, 1024]
        # or the line below with keepdim=False:
        # global_feature = torch.max(x, 2, keepdim=True)[0]       # [B, 1024, 1] because of keepdim=True

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
        self.fc_1 = nn.Linear(1024, 512)                    # weight matrix: [512,1024]        --> output: [B, 512] 
        self.fc_2 = nn.Linear(512, 256)                     # weight matrix: [256,512]         --> output: [B, 256] 
        self.fc_3 = nn.Linear(256, num_classes)             # weight matrix: [num_classes,256] --> output: [B, num_classes] 
        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, 3]

        global_features, ix_maxpool, point_features, feature_tnet, input_tnet = self.base_pointnet(x)
        # [B, 1024] global features vector. Exactly as what nn.Linear expects: [B, 1024]

        x = F.relu(self.bn_1(self.fc_1(global_features)))   # [B, 512]
        x = F.relu(self.bn_2(self.fc_2(x)))                 # [B, 256]
        x = self.dropout(x)
        log_probs = F.log_softmax(self.fc_3(x), dim=1)      # [B, num_classes}]

        return log_probs, ix_maxpool, point_features, feature_tnet, input_tnet



class SegmentationPointNet(nn.Module):
    def __init__(self, num_classes, dropout=0.3, point_dimension=3):
        super().__init__()
        self.base_pointnet = BasePointNet(point_dimension=point_dimension)

        # Shared MLP (512,256,128)
        self.conv_1 = nn.Conv1d(64+1024, 512, 1)          # weight matrix: [512,64+1024] --> output: [B, 512, N] 
        self.conv_2 = nn.Conv1d(512, 256, 1)              # weight matrix: [256,512]     --> output: [B, 256, N] 
        self.conv_3 = nn.Conv1d(256, 128, 1)              # weight matrix: [128,256]     --> output: [B, 128, N] 
        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)
        self.bn_3 = nn.BatchNorm1d(128)
        
        # Shared MLP (128,num_classes)
        self.conv_4 = nn.Conv1d(128, 128, 1)              # weight matrix: [128,128]         --> output: [B, 128, N] 
        self.conv_5 = nn.Conv1d(128, num_classes, 1)      # weight matrix: [num_classes,128] --> output: [B, num_classes, N] 
        self.bn_4 = nn.BatchNorm1d(128)


    def forward(self, x):
        """
        Forward pass for point-wise segmentation.

        INPUT 
        x: [B, N, C_in] (point cloud)
        
        Returns
        log_probs: [B, C_out, N] (per-point log-probabilities)

        Note: The output uses a channels-first layout [B, C_out, N] to be compatible
        with nn.NLLLoss / nn.CrossEntropyLoss, which assume dimension 1 is the
        class (channel) dimension and all remaining dimensions are spatial.
        """
        # x: [B, N, 3]

        global_features, ix_maxpool, point_features, feature_tnet, input_tnet = self.base_pointnet(x)
        # [B, N, 64] point features vector
        # [B, 1024] global features vector
        
        nPoints = point_features.shape[1]
        global_features = global_features.unsqueeze(1)              # [B, 1, 1024]
        global_features = global_features.expand(-1, nPoints , -1)  # [B, N, 1024]
        x =  torch.cat([point_features, global_features], dim=2)    # [B, N, 1024+64]

        x = x.transpose(2, 1)                                       # [B, 1024+64, N]
        x = F.relu(self.bn_1(self.conv_1(x)))                       # [B, 512, N]
        x = F.relu(self.bn_2(self.conv_2(x)))                       # [B, 256, N]
        x = F.relu(self.bn_3(self.conv_3(x)))                       # [B, 128, N]

        x = F.relu(self.bn_4(self.conv_4(x)))                       # [B, 128, N]
        log_probs = F.log_softmax(self.conv_5(x), dim=1)            # [B, num_classes, N]]

        return log_probs, ix_maxpool, point_features, feature_tnet, input_tnet

