import torch
import torch.nn as nn
import torch.optim as optim
import time
from pathlib import Path
from torch.utils.data import DataLoader   
from utils_training import train_model_segmentation, set_seed
from utils_data import load_dataset, info_dataset_batch, choose_architecture
from utils_plotting import plot_metrics, plot_object_parts


# RUN ONLY IF EXECUTED AS MAIN
if __name__ == "__main__":

    # GPU agnostic thingy
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print("Using device:", device)

    config = {"architecture": "SegPointNet",
              "dataset": "ShapeNet",
              "class_name": "Airplane",         # Specific class: "Airplane", "Chair"... | Classification (all classes): "" |
              "nPoints": 1024,
              "seed": 42}  

    hyper = {"batch_size": 32,
             "epochs": 2,
             "lr": 0.001}
        
    # Path creates an object Path and / extends it
    RUN_NAME = f"{config['architecture']}_{config['class_name']}_{config['dataset']}_{config['nPoints']}pts_{hyper['epochs']}epochs_main"
    run_dir = Path("runs") / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)      # create folder: check if parent folder exist, otherwise create it as well
    checkpoint_path = run_dir / "checkpoint.pt"

    # SEEDS i histories varies
    set_seed(config["seed"])

    # LOADING SHAPENET DATASET
    train_dataset, val_dataset, test_dataset, id_to_name = load_dataset(config)

    # DATALOADERS
    train_loader = DataLoader(train_dataset, batch_size=hyper["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=hyper["batch_size"], shuffle=False)
    num_classes = len(id_to_name)
    
    # CHECKS: sizes
    info_dataset_batch(config["dataset"], train_dataset, train_loader, id_to_name)
    print(f"\nTrain size: {len(train_dataset)}")
    print(f"Val size  : {len(val_dataset)}")
    print(f"Test size : {len(test_dataset)}")
    print(f"Class name: {config['class_name']}")
    print(f"id to part: {id_to_name}")  


    plot_object_parts(train_dataset, 0, id_to_name)
 
    # MODEL + OPTIMIZER + LOSS
    network = choose_architecture(config["architecture"], num_classes).to(device)
    optimizer = optim.Adam(network.parameters(), lr=hyper["lr"])
    criterion = nn.NLLLoss(ignore_index=-1)         # when label is -1 (ground truth) do not include it in teh loss

    # TRAINING LOOP
    time_start_traning = time.time()
    metrics = train_model_segmentation(hyper, train_loader, val_loader, network, optimizer, criterion)
    time_training = time.time() - time_start_traning
    print(f"Training time: {time_training}")

    torch.save({"model": network.state_dict(),
                "config": config, 
                "hyper": hyper,
                "metrics": metrics,
                }, checkpoint_path)

    # PLOTTING CURVES
    plot_metrics(metrics, metric="acc", save_dir=run_dir)
    plot_metrics(metrics, metric="miou", save_dir=run_dir)