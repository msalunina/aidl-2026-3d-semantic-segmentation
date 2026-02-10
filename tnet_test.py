import torch
from tnet import TransformationNet
from torch.utils.data import DataLoader
from utils.shapenet_dataset import shapeNetDataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def test_tnet(): 

    dataset_path = "F:/AIDL_FP/Datasets/PartAnnotation/"
    pc_size = 1024
    pc_batch_size = 1

    train_dataset = shapeNetDataset(dataset_path, pc_size, 0, "")
    train_loader = DataLoader(train_dataset, batch_size=pc_batch_size, shuffle=True)

    input_transform = TransformationNet(input_dim=3, output_dim=3) 
    feat_transform = TransformationNet(input_dim=64, output_dim=64) 

    input_transform.to(device)
    

    for i, (pointcloud, pc_class, label, seg_class) in enumerate(train_loader):
        pointcloud = pointcloud.to(device)
        trans_mat = input_transform(pointcloud)

        print(f"batch {i}")
        
        break


if __name__ == "__main__":
    test_tnet()