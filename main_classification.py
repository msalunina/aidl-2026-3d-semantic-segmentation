import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as DataLoaderGeometric
from torch.utils.data import DataLoader as DataLoaderTorch 
from utils_training import train_model, plot_curves, set_seed
from utils_data import load_dataset, info_dataset_batch, choose_architecture


# RUN ONLY IF EXECUTED AS MAIN
if __name__ == "__main__":

    # Cuda agnostic thingy
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config = {"architecture": "ClassPointNet",
              "dataset": "ModelNet",
              "class_name": "",         # All classes: "" | Specific class: "Airplane", "Chair"...
              "nPoints": 1024,
              "seed": 42}  
    
    hyper = {"batch_size": 32,
             "epochs": 1,
             "lr": 0.001}
    
    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/{config['architecture']}_{config['dataset']}_{config['nPoints']}pts_{hyper['epochs']}epochs.pt"

    # SEEDS i histories varies
    set_seed(config["seed"])

    # //////////////////////////////////////////////////////////////
    #                         MODELNET  
    # //////////////////////////////////////////////////////////////
    if config["dataset"] == "ModelNet":

        full_train_dataset, _, test_dataset, id_to_name = load_dataset(config)
        # Split train_dataset into train and validation
        train_size = int(0.8 * len(full_train_dataset))
        val_size   = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)) # reproducibility)
    
        # DATALOADERS
        train_loader = DataLoaderGeometric(train_dataset, batch_size=hyper["batch_size"], shuffle=True) 
        val_loader   = DataLoaderGeometric(val_dataset, batch_size=hyper["batch_size"], shuffle=False) 
        test_loader  = DataLoaderGeometric(test_dataset, batch_size=hyper["batch_size"], shuffle=False)

    # //////////////////////////////////////////////////////////////
    #                         SHAPENET
    # //////////////////////////////////////////////////////////////
    elif config["dataset"] == "ShapeNet":   
        
        train_dataset, val_dataset, test_dataset, id_to_name = load_dataset(config)
        
        # DATALOADERS
        train_loader = DataLoaderTorch(train_dataset, batch_size=hyper["batch_size"], shuffle=True) 
        val_loader   = DataLoaderTorch(val_dataset, batch_size=hyper["batch_size"], shuffle=False) 
        test_loader  = DataLoaderTorch(test_dataset, batch_size=hyper["batch_size"], shuffle=False)
    
    else:  raise TypeError(f"No idea what is dataset {config['dataset']}")
    # //////////////////////////////////////////////////////////////

    # CHECKS
    # info_dataset_batch(config["dataset"], train_dataset, train_loader, id_to_name)
    print(f"\nTrain size: {len(train_dataset)}")
    print(f"Val size  : {len(val_dataset)}")
    print(f"Test size : {len(test_dataset)}")
    print(f"Num classes: {len(id_to_name)}\n{id_to_name}")     
    
    # MODEL + OPTIMIZER + LOSS
    num_classes = len(id_to_name)
    network = choose_architecture(config["architecture"], num_classes).to(device)
    optimizer = optim.Adam(network.parameters(), lr=hyper["lr"])
    criterion = nn.NLLLoss()

    # TRAINING LOOP
    time_start_traning = time.time()
    train_loss, train_acc, val_loss, val_acc = train_model(hyper, train_loader, val_loader, network, optimizer, criterion)
    time_training = time.time() - time_start_traning
    print(f"Training time: {time_training}")

    torch.save({"model": network.state_dict(),
                "config": config, 
                "hyper": hyper,
                }, save_path)

    # PLOTTING CURVES
    plot_curves(train_loss, train_acc, val_loss, val_acc)
    