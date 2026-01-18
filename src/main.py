from test import predict
from train import (
    train_model,
    plot_losses,
    plot_accuracies
)
from models.pointnet import PointNetClassification
from utils.utils import preprocess_dataset
from utils.visualization_utils import (
    visualize_point_cloud_matplotlib,
    visualize_multiple_point_clouds
)
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ModelNet
import torch.optim as optim
import torch
import time
from pathlib import Path


def load_data(config):

    print("Loading data...")

    # Load ModelNet10 dataset
    train_dataset = ModelNet(root=config["data_path"], name='10', train=True)
    test_dataset = ModelNet(root=config["data_path"], name='10', train=False)

    # split train dataset into train and val
    l_data = len(train_dataset)
    train_dataset, val_dataset = random_split(
        train_dataset,
        [round((1 - config["validation_split"]) * l_data),
         round(config["validation_split"] * l_data)],
        generator=torch.Generator().manual_seed(1)
    )

    # Apply point cloud preprocessing (sampling and normalization)
    train_dataset_sampled = preprocess_dataset(
        train_dataset, config["num_points"])
    val_dataset_sampled = preprocess_dataset(val_dataset, config["num_points"])
    test_dataset_sampled = preprocess_dataset(
        test_dataset, config["num_points"])

    train_loader = DataLoader(
        train_dataset_sampled, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset_sampled, batch_size=config["batch_size"], shuffle=False, drop_last=False)
    test_loader = DataLoader(
        test_dataset_sampled, batch_size=config["batch_size"], shuffle=False, drop_last=False)

    return train_dataset_sampled, val_dataset_sampled, test_dataset_sampled, train_loader, val_loader, test_loader


def visualize_examples(dataset, class_names, config, sample_idx=10):

    # Visualize a single point cloud from the training set
    print("Visualizing a single point cloud from the training set (sampled 2048 points)...")

    sample = dataset[sample_idx]
    visualize_point_cloud_matplotlib(
        sample.pos.numpy(),
        label=f'{class_names[sample.y.item()]} (sampled 2048 points)',
        figsize=(10, 8),
        output_path=config["vis_single_example_path"]
    )

    # Visualize one example from each class
    print("Creating visualization grid for all classes (no sampling)...")

    # Find one example from each class in the original training dataset
    full_train_dataset = ModelNet(
        root=config["data_path"], name='10', train=True)
    class_examples = {}

    for i in range(len(full_train_dataset)):
        data = full_train_dataset[i]
        class_idx = data.y.item()
        if class_idx not in class_examples:
            class_examples[class_idx] = data
        if len(class_examples) == 10:
            break

    visualize_multiple_point_clouds(
        point_clouds=[class_examples[i].pos.numpy() for i in range(10)],
        labels=[class_names[i] for i in range(10)],
        n_cols=5,
        figsize=(20, 8),
        output_path=config["vis_all_classes_path"],
        title='ModelNet10 Dataset - Example from Each Class (no sampling)'
    )


if __name__ == "__main__":

    start_time = time.time()

    parent_dir = Path(__file__).resolve().parent.parent

    config = {
        "batch_size": 32,
        "validation_split": 0.2,
        "data_path": parent_dir / 'data' / 'modelnet10',
        "vis_single_example_path": parent_dir / 'figs' / 'modelnet10_single_example.png',
        "vis_all_classes_path": parent_dir / 'figs' / 'modelnet10_all_classes.png',
        "num_classes": 10,
        "num_channels": 3,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "num_epochs": 10,
        "num_points": 1024,
    }

    # ModelNet10 class names
    CLASS_NAMES = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
                   'monitor', 'night_stand', 'sofa', 'table', 'toilet']

    # data
    train_dataset_sampled, val_dataset_sampled, test_dataset_sampled, \
        train_loader, val_loader, test_loader = load_data(config)

    visualize_examples(train_dataset_sampled, CLASS_NAMES, config)

    # model training
    model = PointNetClassification(
        num_classes=config["num_classes"],
        input_channels=config["num_channels"],
        dropout=config["dropout"]
    )
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    training_start_time = time.time()
    train_loss_history, val_loss_history, train_acc_history, val_acc_history = train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        model_save_path=parent_dir / 'model_objects' / 'pointnet.pth'
    )
    training_time = time.time() - training_start_time

    plot_losses(
        train_loss_history,
        val_loss_history,
        save_to_file=parent_dir / 'figs' / 'training_validation_loss.png'
    )
    plot_accuracies(
        train_acc_history,
        val_acc_history,
        save_to_file=parent_dir / 'figs' / 'training_validation_accuracy.png'
    )

    # test set evaluation
    test_predictions = predict(model, test_loader)
    test_labels = torch.cat([data.y for data in test_loader.dataset], dim=0)
    test_accuracy = (test_predictions == test_labels).sum(
    ).item() / test_labels.size(0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    total_time = time.time() - start_time

    print("\n" + "="*60)
    print("EXECUTION TIME SUMMARY")
    print("="*60)
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("="*60)
