import argparse
import torch
import numpy as np
import os
import random
from utils.config_parser import ConfigParser
from torch.utils.data import DataLoader
from utils.shapenet_dataset import shapenetDataset
from tqdm import tqdm

from utils.focal_loss import FocalLoss

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from models.pointnet import PointNetSegmentation

from compute_class_frequencies import compute_focal_weights


from utils.trainer import eval_single_epoch_segmentation, train_single_epoch_segmentation

import matplotlib.pyplot as plt


classes_names = [ 'Airplane', 'Bag', 'Cap','Car', 'Chair', 'Earphone','Guitar', 
        'Knife','Lamp', 'Laptop','Motorbike','Mug','Pistol','Rocket', 'Skateboard','Table']
    
classes_size = [4,2,2,4,4,3,3,2,4,2,6,2,3,3,3,3]

classes_folder = ['02691156','02773838','02954340','02958343',
                    '03001627','03261776','03467517','03624134',
                    '03636649','03642806','03790512','03797390',
                    '03948459','04099429','04225987','04379243']

def set_device():
    if torch.cuda.is_available(): 
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): 
        device = torch.device("mps")
    else: device = torch.device("cpu")
    1
    print(f"\nUsing device: {device}")
    return device


def getLossWeightsForDataset(path):

    
    points_per_class = np.zeros(50)
    for class_id in classes_folder:
        files = list(Path(os.path.join(path, class_id)).glob("*.txt"))
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    label = int(float(line.split(" ")[-1]))
                    points_per_class[label]+=1
        
    #compute the weights compute_focal_weights(points_per_class, points_per_class.sum(), 50)
    weights_per_class = points_per_class.sum() / (50 * points_per_class)
    return weights_per_class
        

def main(config):

    device = set_device()

    base_path = "./data/ShapeNet/raw"
    pc_size = 4096

    #loss_weights = getLossWeightsForDataset(base_path)
    
    """Direct inverse
    loss_weights = np.array([2.78288265e-01, 3.93399671e-01, 9.92885292e-01, 1.37892433e+00,
    5.76229184e+01, 4.55120575e+00, 8.27582472e+00, 2.30545934e+01,
    6.71087918e+00, 4.98647959e+00, 2.21998937e+00, 4.96067403e-01,
    2.54424988e-01, 2.11409835e-01, 4.04915845e-01, 2.72325381e+00,
    8.22109615e+00, 1.92900324e+01, 4.55080628e+01, 5.49545886e+00,
    2.41758346e+00, 6.63925415e-01, 2.09398996e+00, 2.07653581e+00,
    1.82575604e+00, 4.03944003e-01, 2.05516861e+01, 1.27633717e+00,
    1.33946507e+00, 1.54831530e+00, 2.96069923e+01, 4.67660721e+01,
    7.08281898e+00, 1.18995566e+02, 2.04881488e+02, 2.39619002e+00,
    2.87035581e+01, 1.80757393e+00, 1.82148116e+00, 4.25819807e+00,
    2.24646402e+01, 8.03024927e+00, 3.05458382e+01, 5.16889608e+01,
    2.01781131e+01, 2.80746287e+00, 3.28133785e+01, 8.41115039e-02,
    2.67126574e-01, 1.60079651e+00], dtype=np.float32)
    """
    
    loss_weights = np.array([0.17754055, 0.21108912, 0.33534504, 0.39519221, 2.55109139, 0.71790496,
                            0.96798568,  1.61503408, 0.8717059,  0.75144295, 0.50142325, 0.23703806,
                            0.169758,    0.15474386, 0.21415643, 0.55535108, 0.96478101, 1.4774414,
                            2.26779484,  0.78885185, 0.5232601,  0.27422406, 0.48698736, 0.48495372,
                            0.45473112,  0.21389928, 1.52494377, 0.38020864, 0.38949714, 0.4187608,
                            1.82990715,  2.2988541,  0.89552829, 3.66041533, 4.79282232, 0.52094004,
                            1.8018124,   0.4524614,  0.4541985,  0.69441617, 1.59425978, 0.95352145,
                            1.85865059,  2.4165264,  1.51103468, 0.56387087, 1.92629423, 0.09760676,
                            0.17394373,  0.42579819], dtype=np.float32)
    
    dataset_train = shapenetDataset(base_path, "train", pc_size, 0.7)
    dataset_eval = shapenetDataset(base_path, "eval",  pc_size, 0.7)
    dataset_test = shapenetDataset(base_path, "test",  pc_size, 0.0)

    loader_train = DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=True)
    loader_eval = DataLoader(dataset_eval, batch_size=config["batch_size"], shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=config["batch_size"], shuffle=True)

    model = PointNetSegmentation(num_classes=50, # 50 shape classes 
                                input_channels=3, 
                                dropout=config["dropout"]).to(device)


    loss_weights = torch.tensor(loss_weights, dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=loss_weights, gamma=2.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['epochs'],
            eta_min=0.00001
    )
    
    writer = SummaryWriter(log_dir=f"./logs/shapenet_{config['epochs']}_ep")
    
    test_name = f"shapenet_{config['epochs']}"

    for epoch in range(config["epochs"]):
              
        train_loss_epoch, train_acc_epoch, train_miou_epoch, train_iou_class_epoch = train_single_epoch_segmentation(config, loader_train, model, optimizer, criterion)

        val_loss_epoch, val_acc_epoch, val_miou_epoch, val_iou_class_epoch = eval_single_epoch_segmentation(config, loader_eval, model, criterion)
        
        scheduler.step()
        
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

        # Add per-class IoU metrics

        offset = 0
        for i, class_name in enumerate(classes_names):
            t_miou = 0
            v_miou = 0
            for m in range(classes_size[i]):
                t_miou = train_iou_class_epoch[offset].item()
                v_miou = val_iou_class_epoch[offset].item()
                offset+=1
            v_miou /= classes_size[i]
            t_miou /= classes_size[i]

            writer.add_scalar(tag=f"{test_name}_IoU_Class/{class_name}/Train", scalar_value=t_miou, global_step=epoch)
            writer.add_scalar(tag=f"{test_name}_IoU_Class/{class_name}/validaiton", scalar_value=t_miou, global_step=epoch)


    checkpoint = {"model_state_dict": model.cpu().state_dict()}
    path_to_save = os.path.join("./snapshots",f"pointnet_shapenet__{config['epochs']}_epochs.pt")
    torch.save(checkpoint, path_to_save)
    

if __name__ == '__main__':

    config = {
        "epochs" : 100,
        "batch_size": 32,
        "lr": 0.01,
        "dropout": 0.3
        }

    main(config)