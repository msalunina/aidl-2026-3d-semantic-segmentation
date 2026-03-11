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
BEV_SUFFIX = "__bev_full_256x256.npz"

np.random.seed(0)


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
    centroid = points.mean(axis=0)
    pts = points - centroid

    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale

    return pts.astype(np.float32), centroid.astype(np.float64), float(scale)


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
            points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)

            labels_raw = np.array(las.classification, dtype=np.int64)
            labels = remap_labels(labels_raw)

            blocks, x_min_las, y_min_las = tile_xy(points, labels, BLOCK_SIZE, STRIDE)

            if MAX_BLOCKS_PER_TILE is not None and len(blocks) > MAX_BLOCKS_PER_TILE:
                sel = np.random.choice(len(blocks), MAX_BLOCKS_PER_TILE, replace=False)
                blocks = [blocks[i] for i in sel]

            for i, (window_pts, window_lbl, (x0, y0)) in enumerate(blocks):
                n_points_in_window = int(window_pts.shape[0])

                # SAME sampling behavior
                bpts, blbl, sample_idx, sampled_with_replacement = sample_fixed(
                    window_pts, window_lbl, N_POINTS
                )

                # SAME normalization behavior
                bpts_norm, norm_centroid, norm_scale = normalize_block(bpts)

                # Window indices relative to LAS tiling origin
                tile_ix, tile_iy = compute_tile_indices(x0, y0, x_min_las, y_min_las, STRIDE)

                # NEW: deterministic BEV identifier
                bev_key = build_bev_key(base, tile_ix, tile_iy)
                bev_filename = build_bev_filename(bev_key)

                out_path = os.path.join(out_dir, f"{base}_b{i:05d}.npz")

                np.savez_compressed(
                    out_path,

                    # Original fields
                    points=bpts_norm,
                    labels=blbl,

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

                    # NEW: direct BEV linkage
                    bev_key=np.array([bev_key]),
                    bev_filename=np.array([bev_filename]),

                    n_points_in_window=np.int32(n_points_in_window),
                    sample_idx=sample_idx,
                    sampled_with_replacement=sampled_with_replacement,

                    # Normalization metadata
                    norm_centroid_xyz=norm_centroid,
                    norm_scale=np.float64(norm_scale),
                )

        print(f"Done {split} -> {out_dir}")


if __name__ == "__main__":
    main()