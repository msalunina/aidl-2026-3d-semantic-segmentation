import argparse
from datetime import datetime
import torch
import numpy as np
import os
import random
from utils.config_parser import ConfigParser
from utils.dataset import DALESDataset
from torch.utils.data import DataLoader
from utils.trainer import train_model_segmentation
from utils.focal_loss import FocalLoss
from utils.sampler import ClassBalancedSampler
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

    start_time = datetime.now()
    
    base_path = Path(os.getcwd())
    if "src" in base_path.parts:
        base_path = base_path[:-1]
    
    config_parser = ConfigParser(
        default_config_path="config/default.yaml",
        parser=argparse.ArgumentParser(description='3D Semantic Segmentation on DALES Dataset')
    )
    config = config_parser.load()
    config_parser.display()

    use_image = False
    if config.model_name == "ipointnet":
        use_image = True

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

    # set seed
    set_seed(config.dataset_seed)

    # Set device
    device = set_device()

    print("\n" + "="*60)
    print("CREATING DATASETS AND DATALOADERS")
    print("="*60)
    # Create datasets
    train_dataset = DALESDataset(
        data_dir=f"{config.model_data_path}/train",
        images_dir=f"{config.image_data_path}/train",
        split='train',
        use_images=use_image,
        use_features=config.dataset_use_features,
        num_points=config.train_num_points,
        normalize=config.dataset_normalize,
        augmentation=config.dataset_augmentation,
        rotation_deg_max=config.dataset_rotation_deg_max,
        train_ratio=config.dataset_train_ratio,
        val_ratio=config.dataset_val_ratio,
        seed=config.dataset_seed
    )
    val_dataset = DALESDataset(
        data_dir=f"{config.model_data_path}/train",
        images_dir=f"{config.image_data_path}/train",
        split='val',
        use_images=use_image,
        use_features=config.dataset_use_features,
        num_points=config.train_num_points,
        normalize=config.dataset_normalize,
        train_ratio=config.dataset_train_ratio,
        val_ratio=config.dataset_val_ratio,
        seed=config.dataset_seed
    )

    if config.use_sampler:
        # Create DataLoader with class-balanced sampling for training
        print("\nSetting up class-balanced sampling for training...")
        train_sampler = ClassBalancedSampler(
            train_dataset, 
            rare_classes=[3, 4],  # Vehicle and Utility classes
            rare_class_boost=3.0,  # Blocks with rare classes are 3x more likely to be sampled
            verbose=True
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            sampler=train_sampler, # Use sampler instead of shuffle
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
    else:
        # Create standard DataLoader for training
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True, 
            persistent_workers=True
        )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    print("\n" + "="*60)
    print("INITIALIZING MODEL, LOSS and OPTIMIZER")
    print("="*60)
    # Initialize model dependent on the provided input
    if config.model_name == "pointnet":
        from models.pointnet import PointNetSegmentation
        model = PointNetSegmentation(num_classes=config.num_classes, 
                                     input_channels=config.num_channels, 
                                     dropout=config.dropout_rate).to(device)
    elif config.model_name == "ipointnet":
        from models.pointnet import IPointNetSegmentation
        model = IPointNetSegmentation(num_classes=config.num_classes, 
                                      input_channels=config.num_channels, 
                                      dropout=config.dropout_rate).to(device)
    elif config.model_name == "pointnetplusplus":
        from models.pointnetplusplus import PointNetPlusPlusSegmentation
        model = PointNetPlusPlusSegmentation(num_classes=config.num_classes,
                                             extra_channels=config.num_channels - 3,
                                             dropout=config.dropout_rate,
                                             grouping=config.grouping,          
                                             K=config.k_neighbors,                  
                                             radius=config.radius).to(device)   

    else: 
        raise ValueError(f"Model name {config.model_name} does not exist")
        
    # Define loss function and optimizer
    loss_weights = torch.tensor(config.loss_weights, dtype=torch.float32).to(device)
    if config.loss_function == "focal_loss":
        criterion = FocalLoss(alpha=loss_weights, gamma=1.0, ignore_index=config.ignore_label)
        print("Using Focal Loss with gamma=1.0 and class weights")         
    elif config.loss_function == "nll_loss":
        criterion = torch.nn.NLLLoss(weight=loss_weights, ignore_index=config.ignore_label)
        print("Using NLL Loss with class weights")
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"{config.optimizer} needs to be coded, stick to Adam")

    # Learning rate scheduler
    if config.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.num_epochs,
            eta_min=config.scheduler_min_lr
        )
        print(f"Using CosineAnnealingLR scheduler: T_max={config.num_epochs}, eta_min={config.scheduler_min_lr}")

    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    metrics = train_model_segmentation(config, train_loader, val_loader, model, optimizer, criterion, scheduler, device, base_path)

    model_objects_dir = base_path / "model_objects"
    model_objects_dir.mkdir(parents=True, exist_ok=True)
    model_export_path = model_objects_dir / f"{config.test_name}.pt"
    torch.save(model.state_dict(), model_export_path)
    wandb.save(str(model_export_path), base_path=str(base_path))

    if metrics.get("val_acc"):
        wandb.run.summary["val_acc"] = float(max(metrics["val_acc"]))
    if metrics.get("val_loss"):
        wandb.run.summary["val_loss"] = float(min(metrics["val_loss"]))
    if metrics.get("val_miou"):
        wandb.run.summary["val_miou"] = float(max(metrics["val_miou"]))

    wandb.finish()

    end_time = datetime.now()
    print(f"\nTotal training time: {end_time - start_time}")

