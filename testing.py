import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time

from src.utils.shapenet_dataset import shapeNetDataset
from src.models.base_pointnet import BasePointNet

from src.models.classification_pointnet import ClassificationPointNet
from src.models.segmentation_pointnet import SegmentationPointNet

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def testSegmentationNet(loader):
    num_points = 1024
    model = SegmentationPointNet(3,1024,16,0.3).to(device)

    for i, (pointcloud, pc_class, label, seg_class) in enumerate(loader):
     
        pointcloud, pc_class = pointcloud.to(device), pc_class.to(device)
        
        x, tnet_out, ix_maxpool = model(pointcloud)
          
        print(f"output {x.shape}")

def testClassificationNet(loader):
    num_points = 1024
    model = ClassificationPointNet(3,1024,16,0.3).to(device)

    for i, (pointcloud, pc_class, label, seg_class) in enumerate(loader):
     
       
        pointcloud, pc_class = pointcloud.to(device), pc_class.to(device)
        
        x, tnet_out, ix_maxpool = model(pointcloud)
          
        print(f"output {x.shape}")


def testBasePointNet(loader):

    num_points = 1024
    model = BasePointNet(3, num_points).to(device)
    
    for i, (pointcloud, pc_class, label, seg_class) in enumerate(loader):
     
       
        pointcloud, pc_class = pointcloud.to(device), pc_class.to(device)
        
        global_feature_vector, feature_transform, tnet_out, ix_maxpool = model(pointcloud)
        global_feature_expanded = global_feature_vector.repeat(1, 1, num_points)
        global_feature_expanded = global_feature_expanded.reshape(pointcloud.shape[0], global_feature_vector.shape[1], num_points)
        
        x = torch.cat([feature_transform, global_feature_expanded], dim=1)
        
        print(f"feature transform shape {feature_transform.shape}")
        print(f"gloabl feat vector {global_feature_vector.shape}")
        
        
        """
        identity = torch.eye(feature_transform.shape[-1])
        if torch.cuda.is_available():
            identity = identity.cuda()
        regularization_loss = torch.norm(
            identity - torch.bmm(feature_transform, feature_transform.transpose(2, 1)))
        """
        
        
        #start = time.perf_counter()
    

if __name__ == "__main__":
    
    #dataset_path = "/mnt/456c90d8-963b-4daa-a98b-64d03c08e3e1/Black_1TB/datasets/shapenet/PartAnnotation"
    dataset_path = "F:/AIDL_FP/Datasets/PartAnnotation"
    point_cloud_size = 1024
    batch_size_cfg = 1
    train_dataset = shapeNetDataset(dataset_path, point_cloud_size, 0, "")

    train_loader = DataLoader(train_dataset, batch_size=batch_size_cfg, shuffle=True)
    #testBasePointNet(train_loader)
    #testClassificationNet(train_loader)
    testSegmentationNet(train_loader)