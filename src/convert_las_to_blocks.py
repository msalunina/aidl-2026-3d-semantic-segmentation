# ----------------------------------------------------------------
# convert_las_to_blocks_with_XY_BEV_Key_metadata.py
#
# Same as the original convert_las_to_blocks.py, BUT saves metadata
# so we can align a PointNet block to the exact LAS spatial window
# and match the corresponding full-density BEV tile deterministically.
#
# IMPORTANT: Keeps the SAME behavior:
# - LAS labels are 0..8, mapped via dales_label_map.py to 5 classes {0..4}
# - same tiling (50m, 25m stride)
# - same sampling (4096) with np.random.seed(0)
# - same normalization (center + unit sphere)
#
# NEW:
# - saves XY metadata
# - saves tile_ix / tile_iy
# - saves bev_key
# - saves bev_filename
# - saves xy_grid for per-point BEV feature sampling
# ----------------------------------------------------------------

import os
import glob
from pathlib import Path

import numpy as np
import laspy
import yaml
from tqdm import tqdm

from utils.dales_label_map import DALES_TO_SIMPLIFIED, IGNORE_LABEL


def load_yaml_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---- Config (cross-platform, project-relative) ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
cfg = load_yaml_config(CONFIG_PATH)

IN_ROOT = (PROJECT_ROOT / cfg["paths"]["raw_data"]).resolve()
OUT_ROOT = (PROJECT_ROOT / cfg["paths"]["model_data"]).resolve()

# Keep original behavior
BLOCK_SIZE = 50.0
STRIDE = 25.0
N_POINTS = 4096
MIN_POINTS_IN_BLOCK = 1024

# Optional cap for quick experiments
MAX_BLOCKS_PER_TILE = None

# Match current full-density BEV naming convention
BEV_H = 256
BEV_W = 256
BEV_SUFFIX = f"__bev_full_{BEV_H}x{BEV_W}.npz"

# Extract features beyond XYZ if configured (e.g. intensity, return info)
ALL_FEATURES = cfg["data_preprocessing"]["extract_features"]

np.random.seed(0)


def extract_all_features_from_las(las_file):
    """
    Extract all available features from a LAS file based on configuration.

    Returns:
        features: numpy array of shape (n_points, num_selected_features)
        feature_names: list of feature names in order
    """
    features_list = []
    feature_names = []

    # 1. XYZ coordinates (always present)
    xyz = np.vstack((las_file.x, las_file.y, las_file.z)).T.astype(np.float64)
    features_list.append(xyz)
    feature_names.extend(["x", "y", "z"])

    # 2. Intensity
    if "intensity" in ALL_FEATURES:
        if hasattr(las_file, "intensity"):
            intensity = np.array(las_file.intensity, dtype=np.float64).reshape(-1, 1)
            features_list.append(intensity)
            feature_names.append("intensity")
        else:
            print("Warning: intensity not found in LAS file")

    # 3. Return number
    if "return_number" in ALL_FEATURES:
        if hasattr(las_file, "return_number"):
            return_num = np.array(las_file.return_number, dtype=np.float64).reshape(-1, 1)
            features_list.append(return_num)
            feature_names.append("return_number")
        else:
            print("Warning: return_number not found in LAS file")

    # 4. Number of returns
    if "number_of_returns" in ALL_FEATURES:
        if hasattr(las_file, "number_of_returns"):
            num_returns = np.array(las_file.number_of_returns, dtype=np.float64).reshape(-1, 1)
            features_list.append(num_returns)
            feature_names.append("number_of_returns")
        else:
            print("Warning: number_of_returns not found in LAS file")

    # 5. Scan angle rank
    if "scan_angle_rank" in ALL_FEATURES:
        if hasattr(las_file, "scan_angle_rank"):
            scan_angle_rank = np.array(las_file.scan_angle_rank, dtype=np.float64).reshape(-1, 1)
            features_list.append(scan_angle_rank)
            feature_names.append("scan_angle_rank")
        else:
            print("Warning: scan_angle_rank not found in LAS file")

    # Final channel order:
    # [x, y, z, intensity, return_number, number_of_returns, scan_angle_rank]
    features = np.hstack(features_list)
    return features, feature_names


def remap_labels(las_labels: np.ndarray) -> np.ndarray:
    out = np.full(las_labels.shape, IGNORE_LABEL, dtype=np.int64)
    for k, v in DALES_TO_SIMPLIFIED.items():
        out[las_labels == k] = v
    return out


def tile_xy(points: np.ndarray, labels: np.ndarray, block_size: float, stride: float):
    """
    Return:
      blocks: list of (block_points, block_labels, (x0, y0))
      x_min_las, y_min_las: LAS min bounds used to define the tiling origin
    """
    blocks = []
    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    x_max, y_max = points[:, 0].max(), points[:, 1].max()

    # Keep same behavior / same tiling family as before
    x_starts = np.arange(x_min, x_max - block_size, stride)
    y_starts = np.arange(y_min, y_max - block_size, stride)

    for x0 in x_starts:
        for y0 in y_starts:
            mask = (
                (points[:, 0] >= x0) & (points[:, 0] < x0 + block_size) &
                (points[:, 1] >= y0) & (points[:, 1] < y0 + block_size)
            )
            if mask.sum() < MIN_POINTS_IN_BLOCK:
                continue
            blocks.append((points[mask], labels[mask], (float(x0), float(y0))))

    return blocks, float(x_min), float(y_min)


def sample_fixed(points: np.ndarray, labels: np.ndarray, n_points: int):
    """
    SAME sampling logic as original, but also returns metadata.
    Returns:
      sampled_points, sampled_labels, sample_idx, sampled_with_replacement(0/1)
    """
    n = points.shape[0]
    if n >= n_points:
        idx = np.random.choice(n, n_points, replace=False)
        rep = 0
    else:
        idx = np.random.choice(n, n_points, replace=True)
        rep = 1

    return points[idx], labels[idx], idx.astype(np.int32), np.int32(rep)


def normalize_block(points: np.ndarray):
    """
    SAME normalization as original, but also returns centroid/scale.
    """
    points = points.copy()

    # Normalize only XYZ coordinates (first 3 channels)
    xyz = points[:, :3]
    centroid = xyz.mean(axis=0)
    pts = xyz - centroid

    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale

    points[:, :3] = pts.astype(np.float32)
    return points, centroid.astype(np.float64), float(scale)


def compute_tile_indices(x0: float, y0: float, x_min_las: float, y_min_las: float, stride: float):
    """
    Convert window origin to integer indices (for matching BEV filenames).
    Round to avoid float drift.
    """
    ix = int(np.round((x0 - x_min_las) / stride))
    iy = int(np.round((y0 - y_min_las) / stride))
    return ix, iy


def build_bev_key(base: str, tile_ix: int, tile_iy: int) -> str:
    """
    Example:
      5080_54435 + x0000 + y0002
      -> 5080_54435__x0000_y0002
    """
    return f"{base}__x{tile_ix:04d}_y{tile_iy:04d}"


def build_bev_filename(bev_key: str) -> str:
    """
    Example:
      5080_54435__x0000_y0002
      -> 5080_54435__x0000_y0002__bev_full_256x256.npz
    """
    return f"{bev_key}{BEV_SUFFIX}"


def compute_xy_grid(sampled_points_xyzxy: np.ndarray, x0: float, y0: float, block_size_m: float):
    """
    Compute per-point XY coordinates in [-1, 1] for torch grid_sample.

    IMPORTANT:
    - sampled_points_xyzxy must be the FINAL sampled points before normalization
    - uses raw XY coordinates in meters
    - aligned with BEV tiles generated from the same spatial window

    Returns:
        xy_grid: (N, 2) float32
                 [:,0] = x_grid in [-1,1]
                 [:,1] = y_grid in [-1,1]
    """
    x = sampled_points_xyzxy[:, 0]
    y = sampled_points_xyzxy[:, 1]

    # Relative coordinates inside current 50m block
    x_rel = (x - x0) / block_size_m
    y_rel = (y - y0) / block_size_m

    # Clamp just in case of floating-point boundary effects
    x_rel = np.clip(x_rel, 0.0, 1.0)
    y_rel = np.clip(y_rel, 0.0, 1.0)

    # Convert to [-1, 1] to match grid_sample convention
    x_grid = x_rel * 2.0 - 1.0
    y_grid = y_rel * 2.0 - 1.0

    xy_grid = np.stack([x_grid, y_grid], axis=1).astype(np.float32)
    return xy_grid


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    for split in ["train", "test"]:
        in_dir = os.path.join(IN_ROOT, split)
        out_dir = os.path.join(OUT_ROOT, split)
        os.makedirs(out_dir, exist_ok=True)

        las_files = sorted(glob.glob(os.path.join(in_dir, "*.las")))
        if len(las_files) == 0:
            raise RuntimeError(f"No .las files found in {in_dir}")

        print(f"\n=== {split.upper()} ===")
        print("Config:", str(CONFIG_PATH))
        print("Input :", in_dir)
        print("Files :", len(las_files))
        print("Output:", out_dir)

        for las_path in tqdm(las_files, desc=f"Processing {split}"):
            base = os.path.splitext(os.path.basename(las_path))[0]

            las = laspy.read(las_path)

            # Extract ALL features
            features, feature_names = extract_all_features_from_las(las)

            labels_raw = np.array(las.classification, dtype=np.int64)
            labels = remap_labels(labels_raw)

            blocks, x_min_las, y_min_las = tile_xy(features, labels, BLOCK_SIZE, STRIDE)

            if MAX_BLOCKS_PER_TILE is not None and len(blocks) > MAX_BLOCKS_PER_TILE:
                sel = np.random.choice(len(blocks), MAX_BLOCKS_PER_TILE, replace=False)
                blocks = [blocks[i] for i in sel]

            for i, (window_pts, window_lbl, (x0, y0)) in enumerate(blocks):
                n_points_in_window = int(window_pts.shape[0])

                # SAME sampling behavior
                bpts_raw, blbl, sample_idx, sampled_with_replacement = sample_fixed(
                    window_pts, window_lbl, N_POINTS
                )

                # NEW: compute xy_grid from sampled RAW XY before normalization
                xy_grid = compute_xy_grid(
                    sampled_points_xyzxy=bpts_raw,
                    x0=x0,
                    y0=y0,
                    block_size_m=BLOCK_SIZE,
                )

                # SAME normalization behavior
                bpts_norm, norm_centroid, norm_scale = normalize_block(bpts_raw)

                # Window indices relative to LAS tiling origin
                tile_ix, tile_iy = compute_tile_indices(x0, y0, x_min_las, y_min_las, STRIDE)

                # Deterministic BEV identifier
                bev_key = build_bev_key(base, tile_ix, tile_iy)
                bev_filename = build_bev_filename(bev_key)

                out_path = os.path.join(out_dir, f"{base}_b{i:05d}.npz")

                np.savez_compressed(
                    out_path,

                    # Original fields
                    points=bpts_norm.astype(np.float32),
                    labels=blbl.astype(np.int64),

                    # Metadata for alignment
                    las_name=np.array([os.path.basename(las_path)]),
                    split=np.array([split]),
                    block_id=np.int32(i),

                    x0=np.float64(x0),
                    y0=np.float64(y0),
                    block_size_m=np.float32(BLOCK_SIZE),
                    stride_m=np.float32(STRIDE),

                    x_min_las=np.float64(x_min_las),
                    y_min_las=np.float64(y_min_las),
                    tile_ix=np.int32(tile_ix),
                    tile_iy=np.int32(tile_iy),

                    # Direct BEV linkage
                    bev_key=np.array([bev_key]),
                    bev_filename=np.array([bev_filename]),

                    # NEW: per-point coordinates for BEV local feature sampling
                    xy_grid=xy_grid,

                    n_points_in_window=np.int32(n_points_in_window),
                    sample_idx=sample_idx,
                    sampled_with_replacement=sampled_with_replacement,

                    # Normalization metadata
                    norm_centroid_xyz=norm_centroid,
                    norm_scale=np.float64(norm_scale),

                    # Optional metadata
                    feature_names=np.array(feature_names, dtype=object),
                )

        print(f"Done {split} -> {out_dir}")


if __name__ == "__main__":
    main()