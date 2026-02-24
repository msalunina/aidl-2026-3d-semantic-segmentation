import argparse
import torch
import numpy as np
import os
import random
from utils.config_parser import ConfigParser
from torch.utils.data import DataLoader
from utils.shapenet_dataset import shapenetDataset
from tqdm import tqdm

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from models.pointnet import PointNetSegmentation


from utils.trainer import eval_single_epoch_segmentation, train_single_epoch_segmentation



def set_device():
    if torch.cuda.is_available(): 
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): 
        device = torch.device("mps")
    else: device = torch.device("cpu")
    1
    print(f"\nUsing device: {device}")
    return device

def train_single_epoch(loader, model, config):

    max_classes = 0
    min_classes = 10000

    for pc, labels in loader:
        mmin = min(labels[0,:])
        mmax = max(labels[0,:])
        min_classes = min(mmin, min_classes)
        max_classes = max(mmax, max_classes)
    
    print(f"TRAIN min label {min_classes} max label {max_classes}")
        

def eval_single_epoch(loader, model, config):
    max_classes = 0
    min_classes = 10000

    for pc, labels in loader:
        mmin = min(labels[0,:])
        mmax = max(labels[0,:])
        min_classes = min(mmin, min_classes)
        max_classes = max(mmax, max_classes)
    
    print(f"EVAL min label {min_classes} max label {max_classes}")

def test_single_epoch(loader):
    max_classes = 0
    min_classes = 10000

    for pc, labels in loader:
        mmin = min(labels[0,:])
        mmax = max(labels[0,:])
        min_classes = min(mmin, min_classes)
        max_classes = max(mmax, max_classes)
    
    print(f"TEST min label {min_classes} max label {max_classes}")

def main(config):

    device = set_device()

    base_path = "./data/ShapeNet/raw"
    pc_size = 1024
    dataset_train = shapenetDataset(base_path, "train", pc_size)
    dataset_eval = shapenetDataset(base_path, "eval", pc_size)
    dataset_test = shapenetDataset(base_path, "test", pc_size)


    loader_train = DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=True)
    loader_eval = DataLoader(dataset_eval, batch_size=config["batch_size"], shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=config["batch_size"], shuffle=True)


    model = PointNetSegmentation(num_classes=50, # 50 shape classes 
                                input_channels=3, 
                                dropout=config["dropout"]).to(device)

    criterion = torch.nn.NLLLoss()         
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    writer = SummaryWriter(log_dir="./logs/shapenet_25_ep")
    test_name = "shapenet_25"
    for epoch in range(config["epochs"]):
    
        train_loss_epoch, train_acc_epoch, train_miou_epoch = train_single_epoch_segmentation(config, loader_train, model, optimizer, criterion)

        val_loss_epoch, val_acc_epoch, val_miou_epoch = eval_single_epoch_segmentation(config, loader_eval, model, criterion)
        
        tqdm.write(f"Epoch: {epoch+1}/{config['epochs']}"
            f" | loss (train/val) = {train_loss_epoch:.3f}/{val_loss_epoch:.3f}"
            f" | acc (train/val) = {train_acc_epoch:.3f}/{val_acc_epoch:.3f}"
            f" | miou (train/val) = {train_miou_epoch:.3f}/{val_miou_epoch:.3f}")
    
        writer.add_scalar(tag=f"{test_name}_Train/Loss", scalar_value=train_loss_epoch, global_step=epoch)
        writer.add_scalar(tag=f"{test_name}_Train/Accuracy", scalar_value=train_acc_epoch, global_step=epoch)
        writer.add_scalar(tag=f"{test_name}_Train/mIoU", scalar_value=train_miou_epoch, global_step=epoch)
        writer.add_scalar(tag=f"{test_name}_Validation/Loss", scalar_value=val_loss_epoch, global_step=epoch)
        writer.add_scalar(tag=f"{test_name}_Validation/Accuracy", scalar_value=val_acc_epoch, global_step=epoch)
        writer.add_scalar(tag=f"{test_name}_Validation/mIoU", scalar_value=val_miou_epoch, global_step=epoch)

    checkpoint = {"model_state_dict": model.cpu().state_dict()}
    path_to_save = os.path.join("./snapshots",f"pointnet_shapenet__{config['epochs']}_epochs.pt")
    torch.save(checkpoint, path_to_save)
    

if __name__ == '__main__':

    config = {
        "epochs" : 25,
        "batch_size": 64,
        "lr": 0.003,
        "dropout": 0.3
        }

    main(config)