# ----------------------------------------------------------------
# Reads .las from our DALES folder
# Remaps 0–8 labels to our 5 classes (configurable)
# Tiles into 50m×50m windows with 25m stride (configurable)
# Samples 4096 points (configurable)
# Extracts multiple features: XYZ, intensity, return info, etc.
# Normalizes per block
# Saves .npz blocks to configured output path
# ----------------------------------------------------------------

import os
import glob
import argparse
import laspy
import numpy as np
from tqdm import tqdm
from utils.config_parser import ConfigParser
from utils.dales_label_map import DALES_TO_SIMPLIFIED, IGNORE_LABEL
from pathlib import Path

np.random.seed(0)


config_parser = ConfigParser(
    default_config_path="config/default.yaml",
    parser=argparse.ArgumentParser(description='3D Semantic Segmentation on DALES Dataset')
)
config = config_parser.load()
IN_ROOT = Path(config.raw_data_path)
OUT_ROOT = Path(config.model_data_path)
BLOCK_SIZE = config.block_size                    # meters
STRIDE = config.stride                            # meters (50% overlap)
N_POINTS = config.preprocess_num_points           # PointNet input size
MIN_POINTS_IN_BLOCK = config.min_points_in_block  # skip tiny blocks
MAX_BLOCKS_PER_TILE = config.max_blocks_per_tile
NORMALIZE = config.preprocess_normalize
ALL_FEATURES = config.extract_features


def extract_all_features_from_las(las_file):
    """
    Extract all available features from a LAS file based on configuration.
    
    Args:
        las_file: laspy LAS file object
    
    Returns:
        features: numpy array of shape (n_points, num_selected_features)
        feature_names: list of feature names in order
    """
    features_list = []
    feature_names = []
    
    # 1. XYZ coordinates (always present)
    xyz = np.vstack((las_file.x, las_file.y, las_file.z)).T.astype(np.float64)
    features_list.append(xyz)
    feature_names.extend(['x', 'y', 'z'])
    
    # 2. Intensity
    if 'intensity' in ALL_FEATURES:
        if hasattr(las_file, 'intensity'):
            intensity = np.array(las_file.intensity, dtype=np.float64).reshape(-1, 1)
            features_list.append(intensity)
            feature_names.append('intensity')
        else:
            print(f"Warning: intensity not found in LAS file")
    
    # 3. Return number
    if 'return_number' in ALL_FEATURES:
        if hasattr(las_file, 'return_number'):
            return_num = np.array(las_file.return_number, dtype=np.float64).reshape(-1, 1)
            features_list.append(return_num)
            feature_names.append('return_number')
        else:
            print(f"Warning: return_number not found in LAS file")
    
    # 4. Number of returns
    if 'number_of_returns' in ALL_FEATURES:
        if hasattr(las_file, 'number_of_returns'):
            num_returns = np.array(las_file.number_of_returns, dtype=np.float64).reshape(-1, 1)
            features_list.append(num_returns)
            feature_names.append('number_of_returns')
        else:
            print(f"Warning: number_of_returns not found in LAS file")

    # 5. Scan angle rank
    if 'scan_angle_rank' in ALL_FEATURES:
        if hasattr(las_file, 'scan_angle_rank'):
            scan_angle_rank = np.array(las_file.scan_angle_rank, dtype=np.float64).reshape(-1, 1)
            features_list.append(scan_angle_rank)
            feature_names.append('scan_angle_rank')
        else:
            print(f"Warning: scan_angle_rank not found in LAS file")
    
    # Concatenate all features: [x, y, z, intensity, return_num, num_returns, scan_angle_rank] = 7 channels
    features = np.hstack(features_list)
    return features, feature_names


def remap_labels(las_labels: np.ndarray) -> np.ndarray:
    out = np.full(las_labels.shape, IGNORE_LABEL, dtype=np.int64)
    for k, v in DALES_TO_SIMPLIFIED.items():
        out[las_labels == k] = v
    return out


def tile_xy(points: np.ndarray, labels: np.ndarray,
            block_size: float, stride: float):
    """Return list of (block_points, block_labels, (xs, ys))."""
    blocks = []
    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    x_max, y_max = points[:, 0].max(), points[:, 1].max()

    x_starts = np.arange(x_min, x_max - block_size, stride)
    y_starts = np.arange(y_min, y_max - block_size, stride)

    for xs in x_starts:
        for ys in y_starts:
            mask = (
                (points[:, 0] >= xs) & (points[:, 0] < xs + block_size) &
                (points[:, 1] >= ys) & (points[:, 1] < ys + block_size)
            )
            if mask.sum() < MIN_POINTS_IN_BLOCK:
                continue
            blocks.append((points[mask], labels[mask], (xs, ys)))

    return blocks


def sample_fixed(points: np.ndarray, labels: np.ndarray, n_points: int):
    n = points.shape[0]
    if n >= n_points:
        idx = np.random.choice(n, n_points, replace=False)
    else:
        idx = np.random.choice(n, n_points, replace=True)
    return points[idx], labels[idx]


def normalize_block(points: np.ndarray):

    # Normalize only XYZ coordinates (first 3 channels)
    xyz = points[:, :3]

    # center
    centroid = xyz.mean(axis=0)
    xyz = xyz - centroid

    # scale to unit sphere
    scale = np.max(np.linalg.norm(xyz, axis=1))
    if scale > 0:
        xyz = xyz / scale

    points[:, :3] = xyz
    return points


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    for split in ["train", "test"]:
        in_dir = IN_ROOT / split
        out_dir = OUT_ROOT / split
        os.makedirs(out_dir, exist_ok=True)

        las_files = sorted(glob.glob(os.path.join(in_dir, "*.las")))
        if len(las_files) == 0:
            raise RuntimeError(f"No .las files found in {in_dir}")

        print(f"\n=== {split.upper()} ===")
        print("Input:", in_dir)
        print("Files:", len(las_files))
        print("Output:", out_dir)

        for las_path in tqdm(las_files, desc=f"Processing {split}"):
            base = os.path.splitext(os.path.basename(las_path))[0]

            las = laspy.read(las_path)
            
            # Extract ALL features (xyz, intensity, return_number, number_of_returns)
            features, feature_names = extract_all_features_from_las(las)
            
            # Extract labels
            labels_raw = np.array(las.classification, dtype=np.int64)
            labels = remap_labels(labels_raw)

            blocks = tile_xy(features, labels, BLOCK_SIZE, STRIDE)

            # Optional: subsample blocks per tile for speed
            if MAX_BLOCKS_PER_TILE is not None and len(blocks) > MAX_BLOCKS_PER_TILE:
                sel = np.random.choice(len(blocks), MAX_BLOCKS_PER_TILE, replace=False)
                blocks = [blocks[i] for i in sel]

            for i, (block_features, block_labels, (xs, ys)) in enumerate(blocks):

                block_features, block_labels = sample_fixed(block_features, block_labels, N_POINTS)
                if NORMALIZE:
                    block_features = normalize_block(block_features)
                
                # Convert to float32 for efficiency
                block_features = block_features.astype(np.float32)

                out_path = os.path.join(out_dir, f"{base}_b{i:05d}.npz")
                np.savez_compressed(out_path, points=block_features, labels=block_labels)

        print(f"Done {split} -> {out_dir}")


if __name__ == "__main__":
    main()
