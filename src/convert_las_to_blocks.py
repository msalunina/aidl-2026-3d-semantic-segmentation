# ----------------------------------------------------------------
# Reads .las from our DALES folder
# Remaps 0–8 labels to our 5 classes
# Tiles into 50m×50m windows with 25m stride
# Samples 4096 points
# Normalizes per block
# Saves .npz blocks to E:\Dales\pointnet_blocks\{train,test}
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
    # center
    centroid = points.mean(axis=0)
    pts = points - centroid

    # scale to unit sphere
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts.astype(np.float32)


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
            points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
            labels_raw = np.array(las.classification, dtype=np.int64)
            labels = remap_labels(labels_raw)

            blocks = tile_xy(points, labels, BLOCK_SIZE, STRIDE)

            # Optional: subsample blocks per tile for speed
            if MAX_BLOCKS_PER_TILE is not None and len(blocks) > MAX_BLOCKS_PER_TILE:
                sel = np.random.choice(len(blocks), MAX_BLOCKS_PER_TILE, replace=False)
                blocks = [blocks[i] for i in sel]

            for i, (bpts, blbl, (xs, ys)) in enumerate(blocks):
                bpts, blbl = sample_fixed(bpts, blbl, N_POINTS)
                bpts = normalize_block(bpts)

                out_path = os.path.join(out_dir, f"{base}_b{i:05d}.npz")
                np.savez_compressed(out_path, points=bpts, labels=blbl)

        print(f"Done {split} -> {out_dir}")


if __name__ == "__main__":
    main()
