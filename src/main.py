import argparse
import torch
from utils.config_parser import ConfigParser
from utils.dataset import DALESDataset
from torch.utils.data import DataLoader
from utils.trainer import train_model_segmentation


def set_device():
    if torch.cuda.is_available(): 
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): 
        device = torch.device("mps")
    else: device = torch.device("cpu")
    
    print(f"\nUsing device: {device}")
    return device


if __name__ == '__main__':

    # TODO: initiate logging

    config_parser = ConfigParser(
        default_config_path="config/default.yaml",
        parser=argparse.ArgumentParser(description='3D Semantic Segmentation on DALES Dataset')
    )
    config = config_parser.load()
    config_parser.display()

    # Set device
    device = set_device()

    print("\n" + "="*60)
    print("CREATING DATASETS AND DATALOADERS")
    print("="*60)
    # Create datasets
    train_dataset = DALESDataset(
        data_dir=f"{config.model_data_path}/train",
        split='train',
        use_features=config.dataset_use_features,
        num_points=config.train_num_points,
        normalize=config.dataset_normalize,
        train_ratio=config.dataset_train_ratio,
        val_ratio=config.dataset_val_ratio,
        seed=config.dataset_seed
    )
    val_dataset = DALESDataset(
        data_dir=f"{config.model_data_path}/train",
        split='val',
        use_features=config.dataset_use_features,
        num_points=config.train_num_points,
        normalize=config.dataset_normalize,
        train_ratio=config.dataset_train_ratio,
        val_ratio=config.dataset_val_ratio,
        seed=config.dataset_seed
    )
    test_dataset = DALESDataset(
        data_dir=f"{config.model_data_path}/test",
        split='test',
        use_features=config.dataset_use_features,
        num_points=config.train_num_points,
        normalize=config.dataset_normalize,
        use_all_files=config.dataset_test_use_all_files,
        seed=config.dataset_seed
    )

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)


    print("\n" + "="*60)
    print("INITIALIZING MODEL, LOSS and OPTIMIZER")
    print("="*60)
    # Initialize model dependent on the provided input
    if config.model_name == "pointnet":
        from models.pointnet import PointNetSegmentation
        model = PointNetSegmentation(num_classes=config.num_classes, 
                                     input_channels=config.num_channels, 
                                     dropout=config.dropout_rate).to(device)
    else: 
        raise ValueError(f"Model name {config.model_name} does not exist")
        
    # Define loss function and optimizer (should be the same?)
    # We use NLLLoss because 'pointnet' outputs log-probabilities (log_softmax)
    # ignore_index = 1: when label is -1, do not include it in the loss
    criterion = torch.nn.NLLLoss(ignore_index=config.ignore_label)         
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"{config.optimizer} needs to be coded, stick to Adam")

    # TODO: we could put a learning rate scheduler here

    # TODO: training loop, we could put it into a separate class (trainer.py)
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    metrics = train_model_segmentation(config, train_loader, val_loader, model, optimizer, criterion)



    # TODO: evaluation loop, we could put it into a separate class (evaluator.py) (measure metrics on test set)

    # TODO: save logs to log file

    # ? if we want to compare metrics across different runs, we could save them to a CSV file
    # to display on the same plot after (in a separate script)

