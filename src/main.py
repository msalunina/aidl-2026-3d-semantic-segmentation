import argparse
from utils.config_parser import ConfigParser
from utils.dataset import DALESDataset
from torch.utils.data import DataLoader


if __name__ == '__main__':

    # TODO: initiate logging

    config_parser = ConfigParser(
        default_config_path="config/default.yaml",
        parser=argparse.ArgumentParser(description='3D Semantic Segmentation on DALES Dataset')
    )
    config = config_parser.load()
    config_parser.display()


    # TODO: set device

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

    # TODO: initialize model dependent on the provided input
    # e.g.:
    # if args.model == 'pointnet':
    #     from models.pointnet import PointNetSegmentation
    #     model = PointNetSegmentation(...)
    # elif args.model == '...':
    #     ...

    # TODO: define loss function and optimizer (should be the same?)

    # TODO: we could put a learning rate scheduler here

    # TODO: training loop, we could put it into a separate class (trainer.py)

    # TODO: evaluation loop, we could put it into a separate class (evaluator.py) (measure metrics on test set)

    # TODO: save logs to log file

    # ? if we want to compare metrics across different runs, we could save them to a CSV file
    # to display on the same plot after (in a separate script)

