import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from pointnet import ClassificationPointNet

from las_dataset import lasDataset
from modelnet_dataset import modelNetDataset

import time
import os


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#modelnet dict
modelnet10 = {
    "bathub": 0,
    "bed": 1,
    "chair": 2,
    "desk": 3,
    "dresser": 4,
    "monitor": 5,
    "night_stand": 6,
    "sofa": 7,
    "table": 8,
    "toilet": 9
}

def train_single_epoch(model, dataset):

    for data, target in dataset:
        #[B n 3]
        print(target)
        data = data.to(device)
        out = model(data)
        #print(out.shape)
                

def train_las_dataset(config):
    
    model = ClassificationPointNet().to(device)
    train_dataset = lasDataset(config["dataset_path"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    train_single_epoch(model, train_loader)


def train_modelnet10_dataset(config):

    model = ClassificationPointNet(len(modelnet10)).to(device)
    train_dataset = modelNetDataset(config["dataset_path"], modelnet10, 128, True)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    train_single_epoch(model, train_loader)



if __name__ == "__main__":

    config = {
        #"dataset_path": "F:\AIDL_FP\Datasets\TerLidar",
        "dataset_path": "F:\AIDL_FP\Datasets\ModelNet10\ModelNet10",
        "epochs": 1,
        "lr": 1e-3,
        "log_interval": 1000,
        "batch_size": 2
    }

    #train_las_dataset(config)
    train_modelnet10_dataset(config)