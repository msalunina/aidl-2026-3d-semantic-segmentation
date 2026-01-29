import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from test import evaluate_segmentation
from train import (
    train_segmentation_model,
    plot_losses,
    plot_accuracies,
    plot_ious
)
from models.pointnet import PointNetSegmentation
import time
from pathlib import Path
from datetime import datetime
import sys
import json
import matplotlib.pyplot as plt
import keras
import torch
import torch.optim as optim
from utils.utils import TeeOutput
from utils.shapenet_dataset import shapeNetDataset
from torch.utils.data import DataLoader, random_split


def create_global_label_mapping(metadata):
    """
    Create a mapping from (object_class_idx, local_part_idx) to global label.
    
    Args:
        metadata: Dictionary containing class metadata with part labels
    
    Returns:
        class_part_to_global: Dictionary mapping (class_idx, local_part_idx) to global_label
        global_to_class_part: Dictionary mapping global_label to (class_idx, part_name)
        num_global_classes: Total number of global part classes
    """
    class_part_to_global = {}
    global_to_class_part = {}
    global_label = 0
    
    # Iterate through classes in sorted order for consistency
    for class_idx, class_name in enumerate(sorted(metadata.keys())):
        part_names = metadata[class_name]["lables"]  # Note: typo in metadata
        
        # Add "unlabeled" as the first part (index -1 maps to this)
        class_part_to_global[(class_idx, -1)] = global_label
        global_to_class_part[global_label] = (class_idx, "unlabeled")
        global_label += 1
        
        # Map each part index of this class to a global label
        for local_part_idx, part_name in enumerate(part_names):
            class_part_to_global[(class_idx, local_part_idx)] = global_label
            global_to_class_part[global_label] = (class_idx, part_name)
            global_label += 1
    
    print(f"\nCreated global label mapping with {global_label} total part classes (including 'unlabeled' for each object class)")
    print("Class-Part to Global Label mapping:")
    for class_idx, class_name in enumerate(sorted(metadata.keys())):
        part_names = metadata[class_name]["lables"]
        unlabeled_label = class_part_to_global[(class_idx, -1)]
        global_labels = [class_part_to_global[(class_idx, idx)] for idx in range(len(part_names))]
        print(f"  {class_name}: unlabeled -> {unlabeled_label}, parts {part_names} -> global labels {global_labels}")
    
    return class_part_to_global, global_to_class_part, global_label


def download_shapenet_dataset():
    dataset_url = "https://git.io/JiY4i"

    keras.utils.get_file(
        fname="shapenet.zip",
        origin=dataset_url,
        cache_subdir="shapenet",
        hash_algorithm="auto",
        extract=True,
        archive_format="auto",
        cache_dir="data",
    )


def visualize_shapenet_batch(pointcloud, labels, obj_class, output_dir, metadata, class_idx_to_name, num_samples=4):
    """
    Visualize a batch of ShapeNet point clouds and save to files.
    
    Args:
        pointcloud: Tensor of shape (batch_size, num_points, 3)
        labels: Tensor of part labels for each point
        obj_class: List of object class indices (tensors)
        seg_class: List of number of parts per object
        output_dir: Directory to save the visualizations
        metadata: Dictionary containing class metadata with part labels and colors
        class_idx_to_name: Dictionary mapping class index to class name
        num_samples: Number of samples to visualize from the batch
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_to_visualize = min(num_samples, pointcloud.shape[0])
    
    for i in range(num_to_visualize):
        class_idx = obj_class[i].item() if torch.is_tensor(obj_class[i]) else obj_class[i]
        class_name = class_idx_to_name[class_idx]
        part_labels = metadata[class_name]["lables"]  # Note: typo in metadata 'lables'
        part_colors = metadata[class_name]["colors"]
        
        # Get unique labels actually present in this sample
        unique_labels = sorted(set(int(label) for label in labels[i]))
        num_metadata_parts = len(part_labels)
        num_actual_parts = len(unique_labels)
        
        if num_actual_parts > num_metadata_parts:
            print(f"Visualizing {class_name} with {num_actual_parts} parts (metadata incomplete: only {num_metadata_parts} documented)")
        else:
            print(f"Visualizing {class_name} with {num_actual_parts} parts")
        
        # Handle both (batch, num_points, 3) and (batch, 3, num_points) formats
        if pointcloud.shape[1] == 3:
            x = pointcloud[i][0, :].numpy()
            y = pointcloud[i][1, :].numpy()
            z = pointcloud[i][2, :].numpy()
        else:
            x = pointcloud[i][:, 0].numpy()
            y = pointcloud[i][:, 1].numpy()
            z = pointcloud[i][:, 2].numpy()
        
        # Map labels to colors from metadata
        # Unlabeled points (label == -1) get grey color, others get colors from metadata
        color = ['#808080' if int(label) == -1 else part_colors[int(label)] for label in labels[i]]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, c=color, s=1)
        
        # Create legend based on unique labels actually present in this sample
        legend_elements = []
        # Add unlabeled to legend if present
        if -1 in [int(label) for label in labels[i]]:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor='#808080', 
                                     markersize=8,
                                     label='unlabeled'))
        # Add part labels from metadata
        for j, part_name in enumerate(part_labels):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=part_colors[j], 
                                     markersize=8,
                                     label=part_name))
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        ax.set_title(f"ShapeNet: {class_name} ({len(unique_labels)} parts)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        output_path = output_dir / f"shapenet_sample_{i}_{class_name}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to {output_path}")


def visualize_shapenet_grid(pointcloud, labels, obj_class, output_path, metadata, class_idx_to_name, max_samples=8):
    """
    Visualize multiple ShapeNet point clouds in a grid layout.
    
    Args:
        pointcloud: Tensor of shape (batch_size, num_points, 3)
        labels: Tensor of part labels for each point
        obj_class: List of object class indices (tensors)
        seg_class: List of number of parts per object
        output_path: Path to save the visualization
        metadata: Dictionary containing class metadata with part labels and colors
        class_idx_to_name: Dictionary mapping class index to class name
        max_samples: Maximum number of samples to include in grid
    """
    
    num_samples = min(max_samples, pointcloud.shape[0])
    n_cols = min(4, num_samples)
    n_rows = (num_samples + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    
    for i in range(num_samples):
        class_idx = obj_class[i].item() if torch.is_tensor(obj_class[i]) else obj_class[i]
        class_name = class_idx_to_name[class_idx]
        part_colors = metadata[class_name]["colors"]
        part_labels = metadata[class_name]["lables"]  # Note: typo in metadata 'lables'
        
        # Handle both (batch, num_points, 3) and (batch, 3, num_points) formats
        if pointcloud.shape[1] == 3:
            x = pointcloud[i][0, :].numpy()
            y = pointcloud[i][1, :].numpy()
            z = pointcloud[i][2, :].numpy()
        else:
            x = pointcloud[i][:, 0].numpy()
            y = pointcloud[i][:, 1].numpy()
            z = pointcloud[i][:, 2].numpy()
        
        # Map labels to colors from metadata
        # Unlabeled points (label == -1) get grey color, others get colors from metadata
        color = ['#808080' if int(label) == -1 else part_colors[int(label)] for label in labels[i]]
        
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        ax.scatter(x, y, z, c=color, s=1)
        # Use generic title since metadata part names may not match actual label assignment
        ax.set_title(f"{class_name}\n({len(set(int(l) for l in labels[i]))} parts)", fontsize=9)
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)
        ax.tick_params(labelsize=6)
    
    plt.suptitle('ShapeNet Dataset - Part Segmentation Examples', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved grid visualization to {output_path}")


if __name__ == "__main__":

    start_time = time.time()

    parent_dir = Path(__file__).resolve().parent.parent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    config = {
        "download_shapenet": False,
        "data_path": parent_dir / 'data' / 'shapenet' / 'shapenet_extracted' / 'PartAnnotation',
        "num_points": 1024,
        "batch_size": 32,
        "validation_split": 0.2,
        "test_split": 0.1,
        "vis_output_dir": parent_dir / 'figs' / 'shapenet_samples',
        "vis_grid_path": parent_dir / 'figs' / 'shapenet_grid.png',
        "num_classes": 42,  # Total number of part classes across all object categories
        "num_channels": 3,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "num_epochs": 30,
        "alpha": 0.001,
        "model_save_path": parent_dir / 'model_objects' / 'pointnet_segmentation.pth',
        "logs_path": parent_dir / 'logs' / f'shapenet_training_{timestamp}.log',
    }

    tee = None
    if config["logs_path"]:
        output_path = Path(config["logs_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tee = TeeOutput(output_path)
        sys.stdout = tee

    # Check GPU availability
    print("="*60)
    print("DEVICE INFORMATION")
    print("="*60)
    if torch.cuda.is_available():
        print(f"CUDA is available!")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("WARNING: CUDA is not available. Training will be VERY slow on CPU!")
    print("="*60)

    if config["download_shapenet"]:
        download_shapenet_dataset()

    # Load metadata
    metadata_path = config["data_path"] / "metadata.json"
    print(f"\nLoading metadata from {metadata_path}...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"Found {len(metadata)} object classes: {', '.join(metadata.keys())}")

    # Create index to class name mapping (sorted for consistency)
    class_idx_to_name = {i: name for i, name in enumerate(sorted(metadata.keys()))}
    
    # Create global label mapping from (class, part) to unique global labels
    class_part_to_global, global_to_class_part, num_global_classes = create_global_label_mapping(metadata)
    
    # Update config with actual number of global classes
    config["num_classes"] = num_global_classes
    print(f"\nUpdated num_classes to {num_global_classes} (total unique parts across all object classes)")

    print("\nLoading ShapeNet dataset...")
    dataset_full = shapeNetDataset(
        config["data_path"], config["num_points"], mode=0, class_name="", 
        class_part_to_global=class_part_to_global)

    # Split into train, val and test sets
    l_data = len(dataset_full)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset_full,
        [round((1 - config["validation_split"] - config["test_split"]) * l_data),
         round(config["validation_split"] * l_data),
         round(config["test_split"] * l_data)],
        generator=torch.Generator().manual_seed(1)
    )

    num_workers = 4 if torch.cuda.is_available() else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    print("\nGenerating visualizations...")
    for pointcloud, pc_class, label, seg_class in train_loader:
        visualize_shapenet_batch(
            pointcloud,
            label,
            pc_class,
            config["vis_output_dir"],
            metadata,
            class_idx_to_name,
            num_samples=4
        )
        visualize_shapenet_grid(
            pointcloud,
            label,
            pc_class,
            config["vis_grid_path"],
            metadata,
            class_idx_to_name,
            max_samples=8
        )

        # Only visualize the first batch
        break

    # Model training
    print("\n" + "="*60)
    print("TRAINING SEGMENTATION MODEL")
    print("="*60)

    model = PointNetSegmentation(
        num_classes=config["num_classes"],
        input_channels=config["num_channels"],
        dropout=config["dropout"]
    )
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    training_start_time = time.time()
    train_loss_history, val_loss_history, train_acc_history, val_acc_history, train_iou_history, val_iou_history = train_segmentation_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        model_save_path=config["model_save_path"]
    )
    training_time = time.time() - training_start_time

    # Plot training metrics
    plot_losses(
        train_loss_history,
        val_loss_history,
        save_to_file=parent_dir / 'figs' / 'shapenet_training_validation_loss.png'
    )
    plot_accuracies(
        train_acc_history,
        val_acc_history,
        save_to_file=parent_dir / 'figs' / 'shapenet_training_validation_accuracy.png'
    )
    plot_ious(
        train_iou_history,
        val_iou_history,
        save_to_file=parent_dir / 'figs' / 'shapenet_training_validation_iou.png'
    )

    # Test set evaluation
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)

    # Load best model
    model.load_state_dict(torch.load(config["model_save_path"]))

    test_accuracy, test_iou = evaluate_segmentation(
        model, test_loader, config["num_classes"])
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Mean IoU: {test_iou:.4f}")

    total_time = time.time() - start_time

    print(f"\nVisualizations saved to {parent_dir / 'figs'}")
    print("\n" + "="*60)
    print("EXECUTION TIME SUMMARY")
    print("="*60)
    print(
        f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(
        f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("="*60)

    # Restore stdout and close file
    if tee:
        sys.stdout = tee.stdout
        tee.close()
        print(
            f"\nTraining log saved to: {Path(config['logs_path']).absolute()}")
