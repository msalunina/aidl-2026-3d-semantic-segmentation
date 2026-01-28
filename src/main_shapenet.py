import torch
from torch.utils.data import DataLoader
from utils.shapenet_dataset import shapeNetDataset
import torch.nn as nn
import torch.nn.functional as F
import keras
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

from pathlib import Path
import time


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


def visualize_shapenet_batch(pointcloud, labels, obj_class, seg_class, output_dir, metadata, class_idx_to_name, num_samples=4):
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
        # Labels are 1-indexed, so subtract 1 to get the correct index
        color = [part_colors[(int(label)-1) % len(part_colors)] for label in labels[i]]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, c=color, s=1)
        
        # Create legend based on unique labels actually present in this sample
        # Labels are 1-indexed, so subtract 1 to get metadata index
        legend_elements = []
        for i, part_name in enumerate(part_labels):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=part_colors[i], 
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


def visualize_shapenet_grid(pointcloud, labels, obj_class, seg_class, output_path, metadata, class_idx_to_name, max_samples=8):
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
        # Labels are 1-indexed, so subtract 1 to get the correct index
        color = [part_colors[(int(label) - 1) % len(part_colors)] for label in labels[i]]
        
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

    config = {
        "download_shapenet": False,
        "data_path": parent_dir / 'data' / 'shapenet' / 'shapenet_extracted' / 'PartAnnotation',
        "num_points": 1024,
        "batch_size": 32,
        "vis_output_dir": parent_dir / 'figs' / 'shapenet_samples',
        "vis_grid_path": parent_dir / 'figs' / 'shapenet_grid.png',
    }
    
    if config["download_shapenet"]:
        download_shapenet_dataset()

    # Load metadata
    metadata_path = config["data_path"] / "metadata.json"
    print(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"Found {len(metadata)} object classes: {', '.join(metadata.keys())}")
    
    # Create index to class name mapping
    class_idx_to_name = {i: name for i, name in enumerate(metadata.keys())}

    print("\nLoading ShapeNet dataset...")
    dataset = shapeNetDataset(config["data_path"], config["num_points"], mode=0, class_name="")
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)

    print("\nGenerating visualizations...")
    for pointcloud, pc_class, label, seg_class in loader:
        visualize_shapenet_batch(
            pointcloud, 
            label, 
            pc_class, 
            seg_class, 
            config["vis_output_dir"],
            metadata,
            class_idx_to_name,
            num_samples=4
        )
        visualize_shapenet_grid(
            pointcloud,
            label,
            pc_class,
            seg_class,
            config["vis_grid_path"],
            metadata,
            class_idx_to_name,
            max_samples=8
        )
        
        # Only visualize the first batch
        break

    total_time = time.time() - start_time
    
    print(f"\nVisualizations saved to {parent_dir / 'figs'}")
    print("\n" + "="*60)
    print("EXECUTION TIME SUMMARY")
    print("="*60)
    # print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("="*60)