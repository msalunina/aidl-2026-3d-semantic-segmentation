import torch
import os
import json
from pathlib import Path
import numpy as np
import random
import math

class shapeNetDataset(torch.utils.data.Dataset):
    
    # dataset_path: path to the PartAnnotation folder, where the metadata.json is located
    # point_cloud_size: number of points in the returned point cloud
    # mode: train=0, val=1, test=2
    # class_name: single class name or "" for all classes
    def __init__(self, dataset_path: str, point_cloud_size: int, mode: int, class_name: str):
        super().__init__()
        self._xy_th = 0.05
        self._z_th = 0.05
        self._dataset_path = dataset_path
        self._target_points = point_cloud_size
        self._mode = mode
        self._dataset = []
        self._class_name = class_name

        self.loadDataset()

    def readJsonMetadata(self, metadata_file: str):
        self._object_classes = {}
        self._metadata = {}
        with open(metadata_file, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)
            self._object_classes = {name: i for i, name in enumerate(self._metadata)}

    def loadDataset(self):
        self.readJsonMetadata(os.path.join(self._dataset_path, "metadata.json"))

        for class_name in self._object_classes:
            if self._class_name and self._class_name != class_name:
                continue

            class_path = self._metadata[class_name]["directory"]

            class_dir = Path(os.path.join(self._dataset_path, class_path, "points"))
            point_files = list(class_dir.glob("*.pts"))
            pc_names = [p.name.split(".")[0] for p in point_files]

            label_dir = Path(os.path.join(self._dataset_path, class_path, "expert_verified", "points_label"))
            label_files = list(label_dir.glob("*.seg"))

            tsize = len(label_files)
            train = int(0.7 * tsize)
            val = int(0.9 * tsize)

            if self._mode == 0:
                labels = label_files[:train]
            elif self._mode == 1:
                labels = label_files[train:val]
            else:
                labels = label_files[val:]

            for file in labels:
                file_name = file.name.split(".")[0]
                if file_name in pc_names:
                    self._dataset.append({
                        "points": os.path.join(self._dataset_path, class_path, "points", file_name + ".pts"),
                        "labels": os.path.join(
                            self._dataset_path, class_path, "expert_verified", "points_label", file_name + ".seg"
                        ),
                        "class": self._object_classes[class_name],
                        "seg_class": len(self._metadata[class_name]["lables"])  # dataset typo
                    })

    def nearPoint(self, pt1, pt2, pt1_class, pt2_class):
        if pt1_class != pt2_class:
            return False
        distxy = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
        distz = abs(pt1[2] - pt2[2])
        return (distxy < self._xy_th) and (distz < self._z_th)

    def downsamplePointCloud(self, point_cloud, labels):
        if self._target_points > len(point_cloud):
            return point_cloud, labels

        points_to_remove = len(point_cloud) - self._target_points
        removed_pos = []

        while len(removed_pos) < points_to_remove:
            idx = random.randrange(len(point_cloud))
            if idx in removed_pos:
                continue

            nei = 0
            for i in range(len(point_cloud)):
                if i == idx:
                    continue
                if self.nearPoint(point_cloud[idx], point_cloud[i], labels[idx], labels[i]) and i not in removed_pos:
                    nei += 1
                if nei >= 2:
                    removed_pos.append(idx)
                    break
            else:
                removed_pos.append(idx)

        pc = [v for i, v in enumerate(point_cloud) if i not in removed_pos]
        lb = [v for i, v in enumerate(labels) if i not in removed_pos]
        return pc, lb

    def interpolatePointCloud(self, point_cloud, labels):
        if self._target_points < len(point_cloud):
            return point_cloud, labels

        pc = list(point_cloud)
        lb = list(labels)
        needed = self._target_points - len(point_cloud)

        while needed > 0:
            i = random.randrange(len(point_cloud))
            for j in range(len(point_cloud)):
                if i != j and self.nearPoint(point_cloud[i], point_cloud[j], labels[i], labels[j]):
                    pc.append([
                        point_cloud[i][0] + self._xy_th / 2,
                        point_cloud[i][1] + self._xy_th / 2,
                        point_cloud[i][2] + self._z_th / 2
                    ])
                    lb.append(labels[j])
                    needed -= 1
                    if needed == 0:
                        break
        return pc, lb

    def readPointCloud(self, file_name, seg_file):
        pc = []
        labels = []
        with open(file_name) as f:
            for line in f:
                x, y, z = map(float, line.split())
                pc.append([x, y, z])
        with open(seg_file) as f:
            for line in f:
                labels.append(int(line) - 1)
        return pc, labels

    def __len__(self):
        return len(self._dataset)

    # ------------------------------------------------------------------
    # ✅ MODIFIED SECTION (dtype + transpose)
    # ------------------------------------------------------------------
    def __getitem__(self, index):
        item = self._dataset[index]
        point_cloud, labels = self.readPointCloud(item["points"], item["labels"])

        if len(point_cloud) > self._target_points:
            point_cloud, labels = self.downsamplePointCloud(point_cloud, labels)
        elif len(point_cloud) < self._target_points:
            point_cloud, labels = self.interpolatePointCloud(point_cloud, labels)

        point_cloud = np.array(point_cloud, dtype=np.float32).transpose(1, 0)  # (3, N)
        labels = np.array(labels, dtype=np.int64)

        return point_cloud, item["class"], labels, item["seg_class"]
