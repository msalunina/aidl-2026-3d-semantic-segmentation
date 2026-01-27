import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
from pathlib import Path
import numpy as np

from shapenet_dataset import shapeNetDataset
from torch.utils.data import DataLoader


# -----------------------------------------------------------------------------
# Robust label -> color mapping (avoids IndexError when labels > len(class_color)-1)
# -----------------------------------------------------------------------------
CMAP = plt.get_cmap("tab20")  # 20 distinct-ish colors


def labels_to_colors(labels):
    """
    labels: 1D iterable of ints length N
    returns: list of RGBA colors length N
    """
    return [CMAP(int(l) % CMAP.N) for l in labels]


# -----------------------------------------------------------------------------
# search for close points
# -----------------------------------------------------------------------------
def nearPoint(pt1: list, pt2: list, pt1_class: int, pt2_class: int, th1: float, th2: float):
    if pt1_class != pt2_class:
        return False

    distxy = math.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))
    distz = abs(pt1[2] - pt2[2])
    return (distxy < th1) and (distz < th2)


# -----------------------------------------------------------------------------
# downsample pointcloud
# -----------------------------------------------------------------------------
def donwsamplePointCloud(point_cloud: list, labels: list, target_points: int, xy_th: float, z_th: float):
    point_cloud_size = len(point_cloud)

    if target_points > point_cloud_size:
        print(f"Error, cannot downsample {point_cloud_size} to {target_points}")
        return point_cloud, labels

    points_to_remove = point_cloud_size - target_points
    removed_pos = []

    while len(removed_pos) < points_to_remove:
        index = random.randrange(len(point_cloud))
        if index in removed_pos:
            continue

        removed = False
        nei = 0
        for i in range(len(point_cloud)):
            if i == index:
                continue

            if nearPoint(point_cloud[index], point_cloud[i], labels[index], labels[i], xy_th, z_th) and i not in removed_pos:
                nei += 1

            if nei >= 2:
                removed_pos.append(index)
                removed = True
                break

        if not removed:
            removed_pos.append(index)

    downsampled_pc = [v for i, v in enumerate(point_cloud) if i not in removed_pos]
    downsampled_labels = [v for i, v in enumerate(labels) if i not in removed_pos]
    return downsampled_pc, downsampled_labels


# -----------------------------------------------------------------------------
# interpolate pointcloud
# -----------------------------------------------------------------------------
def interpolatePointcloud(point_cloud: list, labels: list, target_points: int, xy_th: float, z_th: float):
    point_cloud_size = len(point_cloud)

    if target_points < point_cloud_size:
        print(f"Error, cannot interpolate {point_cloud_size} to {target_points}")
        return point_cloud, labels

    interpolated_pc = point_cloud
    interpolated_labels = labels
    points_to_add = target_points - point_cloud_size

    interpolated_points = []
    interpolated_index = []
    added_labels = []
    added_points = 0

    while added_points < points_to_add:
        if len(interpolated_index) == len(interpolated_pc):
            interpolated_pc.extend(interpolated_points)
            interpolated_labels.extend(added_labels)
            interpolated_points = []
            interpolated_index = []
            added_labels = []

        index = random.randrange(len(point_cloud))
        if index in interpolated_index:
            continue

        for i in range(len(point_cloud)):
            if i == index:
                continue

            if nearPoint(point_cloud[index], point_cloud[i], labels[index], labels[i], xy_th, z_th):
                point = [
                    point_cloud[index][0] + (xy_th / 2.0),
                    point_cloud[index][1] + (xy_th / 2.0),
                    point_cloud[index][2] + (z_th / 2.0),
                ]
                interpolated_points.append(point)
                added_labels.append(labels[i])
                added_points += 1

                if index not in interpolated_index:
                    interpolated_index.append(index)

                if added_points == points_to_add:
                    break

    interpolated_pc.extend(interpolated_points)
    interpolated_labels.extend(added_labels)
    return interpolated_pc, interpolated_labels


# -----------------------------------------------------------------------------
# metadata + file reading
# -----------------------------------------------------------------------------
def readJsonMetadata(metadata_file: str):
    with open(metadata_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        object_classes = {name: i for i, name in enumerate(data)}
    return data, object_classes


def readPointCloud(file_name: str, seg_file: str):
    point_cloud = []
    seg_class = []

    with open(file_name, "r") as f:
        for line in f:
            values = line.split(" ")
            point = [float(values[0]), float(values[1]), float(values[2])]
            point_cloud.append(point)

    with open(seg_file, "r") as f:
        for line in f:
            seg_class.append(int(line) - 1)

    return point_cloud, seg_class


# -----------------------------------------------------------------------------
# single-sample visualization (file based)
# -----------------------------------------------------------------------------
def showPointCloud(path: str):
    metadata, _ = readJsonMetadata(os.path.join(path, "metadata.json"))

    airplane_path = metadata["Airplane"]["directory"]
    data_dir = Path(os.path.join(path, airplane_path, "points"))
    point_files = list(data_dir.glob("*.pts"))

    label_path = Path(os.path.join(path, airplane_path, "expert_verified", "points_label"))
    label_files = list(label_path.glob("*.seg"))

    pc_files = [p.name.split(".")[0] for p in point_files]

    target_points = 4048

    for file in label_files:
        file_name = file.name.split(".")[0]
        index = pc_files.index(file_name)

        point_cloud, seg_labels = readPointCloud(point_files[index], file)
        print(f"Pointcloud {file_name} with points {len(point_cloud)}")

        x = [p[0] for p in point_cloud]
        y = [p[1] for p in point_cloud]
        z = [p[2] for p in point_cloud]
        color = labels_to_colors(seg_labels)

        target_pc = point_cloud
        target_labels = seg_labels
        applied = "Target PointCloud"

        if len(point_cloud) > target_points:
            target_pc, target_labels = donwsamplePointCloud(point_cloud, seg_labels, target_points, 0.05, 0.05)
            applied += " - downsampled"
        elif len(point_cloud) < target_points:
            target_pc, target_labels = interpolatePointcloud(point_cloud, seg_labels, target_points, 0.05, 0.05)
            applied += " - interpolated"

        tx = [p[0] for p in target_pc]
        ty = [p[1] for p in target_pc]
        tz = [p[2] for p in target_pc]
        tcolor = labels_to_colors(target_labels)

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.scatter(x, y, z, c=color, s=8)
        ax.set_title("original pointcloud")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.scatter(tx, ty, tz, c=tcolor, s=8)
        ax2.set_title(applied)
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")

        plt.tight_layout()
        plt.show()
        # break


# -----------------------------------------------------------------------------
# batch visualization (DataLoader based)  <-- FIXED FOR (B, 3, N)
# -----------------------------------------------------------------------------
def showBatchPointcloud(pointcloud, labels, obj_class, seg_class):
    """
    pointcloud: (B, 3, N)   (after your dataset transpose)
    labels:     (B, N)
    """
    # If tensors, convert to numpy for plotting
    if hasattr(pointcloud, "detach"):
        pointcloud = pointcloud.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()

    for i in range(pointcloud.shape[0]):
        print(f"showing {int(obj_class[i])} with {int(seg_class[i])} parts")

        # pointcloud[i] is (3, N)
        x = pointcloud[i][0, :]
        y = pointcloud[i][1, :]
        z = pointcloud[i][2, :]

        color = labels_to_colors(labels[i])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, c=color, s=8)
        ax.set_title("pointcloud (colored by part label)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.tight_layout()
        plt.show()


def testDataLoader(config):
    train_dataset = shapeNetDataset(config["dataset_path"], config["point_cloud_size"], 0, "")
    test_dataset = shapeNetDataset(config["dataset_path"], config["point_cloud_size"], 1, "")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)

    for pointcloud, pc_class, label, seg_class in train_loader:
        showBatchPointcloud(pointcloud, label, pc_class, seg_class)


if __name__ == "__main__":
    config = {
        "dataset_path": r"c:\aidl\Project\shapenet\PartAnnotation",
        "point_cloud_size": 1024,
        "epochs": 1,
        "lr": 1e-3,
        "log_interval": 1000,
        "batch_size": 1,
    }

    # showPointCloud(config["dataset_path"])
    testDataLoader(config)
