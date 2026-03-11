import argparse
from datetime import datetime
import torch
import numpy as np
import os
import random
from utils.config_parser import ConfigParser
from utils.dataset import DALESDataset
from torch.utils.data import DataLoader
from utils.evaluator import test_model_segmentation
from pathlib import Path
import wandb

def set_device():
    if torch.cuda.is_available(): 
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): 
        device = torch.device("mps")
    else: device = torch.device("cpu")
    
    print(f"\nUsing device: {device}")
    return device


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    
    base_path = Path(os.getcwd())
    if "src" in base_path.parts:
        base_path = base_path[:-1]
    
    config_parser = ConfigParser(
        default_config_path="config/default.yaml",
        parser=argparse.ArgumentParser(description='3D Semantic Segmentation on DALES Dataset')
    )
    config = config_parser.load()
    config_parser.display()

    # Initialize W&B
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{config.test_name}_{timestamp}"
    wandb_mode = getattr(config, 'wandb_mode', 'online') if getattr(config, 'wandb_enabled', True) else 'disabled'
    wandb.init(
        project=getattr(config, 'wandb_project', 'aidl-3d-semantic-segmentation'),
        entity=getattr(config, 'wandb_entity', None),
        name=run_name,
        group=config.test_name,
        config=vars(config),
        mode=wandb_mode,
    )

    # set seed (here? config.dataset_seed?)
    set_seed(config.dataset_seed)

    # Set device
    device = set_device()

    print("\n" + "="*60)
    print("CREATING DATASETS AND DATALOADERS")
    print("="*60)
    # Create test datasets
    test_dataset = DALESDataset(
        data_dir=f"{config.model_data_path}/test",
        images_dir=f"{config.image_data_path}/test",
        split='test',
        use_features=config.dataset_use_features,
        num_points=config.train_num_points,
        normalize=config.dataset_normalize,
        use_all_files=config.dataset_test_use_all_files,
        seed=config.dataset_seed
    )

    # Create test DataLoader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    print("\n" + "="*60)
    print("INITIALIZING MODEL, LOSS and OPTIMIZER")
    print("="*60)
    # Initialize model dependent on the provided input
    if config.model_name == "pointnet":
        from models.pointnet import PointNetSegmentation
        model_trained = PointNetSegmentation(num_classes=config.num_classes, 
                                             input_channels=config.num_channels, 
                                             dropout=config.dropout_rate).to(device)
    elif config.model_name == "ipointnet":
        from models.pointnet import IPointNetSegmentation
        model_trained = IPointNetSegmentation(num_classes=config.num_classes, 
                                              input_channels=config.num_channels, 
                                              dropout=config.dropout_rate).to(device)
    else: 
        raise ValueError(f"Model name {config.model_name} does not exist")
    
    epoch=0
    load_dir= os.path.join(base_path, "model_objects", f"{config.test_name}.pt")
    # checkpoint_path = os.path.join(load_dir, f"pointnet_{epoch}_epochs.pt")
    checkpoint = torch.load(load_dir, map_location=device, weights_only=False)    # it loads more things that weights

    # UPDATE ARCHITECTURE
    model_trained.load_state_dict(checkpoint)
    model_trained.to(device)
    model_trained.eval()      # changes behaviour of some layers (e.g. dropout off, batchnorm), does not strop gradient

        
    # Define loss function and optimizer (should be the same?)
    # We use NLLLoss because 'pointnet' outputs log-probabilities (log_softmax)
    # ignore_index = -1: when label is -1, do not include it in the loss
    loss_weights = torch.tensor(config.loss_weights, dtype=torch.float32).to(device)
    criterion = torch.nn.NLLLoss(weight=loss_weights, ignore_index=config.ignore_label)         


    # TODO: measure metrics on test set
    # Testing loop, we could put it into a separate class (trainer.py)
    print("\n" + "="*60)
    print("TESTING")
    print("="*60)
    with torch.no_grad():       # Stops tracking gradients, saves memory
        metrics = test_model_segmentation(config, test_loader, model_trained, criterion, device, base_path)

    wandb.finish()

