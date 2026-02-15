"""
generate_block_bev_rasters.py

Generate BEV raster representations (256x256) for each normalized
PointNet block stored as .npz.

Input:
    config.model_data_path/{train,test}/*.npz

Output:
    config.model_data_path_BEV/{train,test}/*.npz

Each output file contains 4 BEV channels:
    - density  : log(1 + count)
    - z_max    : max Z in cell
    - z_mean   : mean Z in cell
    - z_range  : z_max - z_min
"""

import glob
import argparse
import numpy as np
from pathlib import Path

# ---- Config import ----
from utils.config_parser import ConfigParser

# ---- BEV resolution ----
H = 256
W = 256


def to_grid_xy(points, H, W):
    """
    Map normalized XY [-1,1] to pixel indices.
    """
    x = points[:, 0]
    y = points[:, 1]

    u = ((x + 1) / 2 * (W - 1)).astype(np.int32)
    v = ((y + 1) / 2 * (H - 1)).astype(np.int32)

    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)

    return u, v


def rasterize(points, H, W):
    """
    Create 4-channel BEV raster.
    """
    u, v = to_grid_xy(points, H, W)
    z = points[:, 2]

    density = np.zeros((H, W), dtype=np.float32)
    z_max   = np.full((H, W), -np.inf, dtype=np.float32)
    z_min   = np.full((H, W), +np.inf, dtype=np.float32)
    z_sum   = np.zeros((H, W), dtype=np.float32)

    for i in range(points.shape[0]):
        ui, vi = u[i], v[i]
        density[vi, ui] += 1
        z_sum[vi, ui] += z[i]
        z_max[vi, ui] = max(z_max[vi, ui], z[i])
        z_min[vi, ui] = min(z_min[vi, ui], z[i])

    mask = density > 0

    z_mean = np.zeros_like(density)
    z_mean[mask] = z_sum[mask] / density[mask]

    z_range = np.zeros_like(density)
    z_range[mask] = z_max[mask] - z_min[mask]

    density = np.log1p(density)
    z_max[~mask] = 0.0

    return density, z_max, z_mean, z_range


def main():

    # ---- Load config ----
    config_parser = ConfigParser(
        default_config_path="config/default.yaml",
        parser=argparse.ArgumentParser(
            description="Generate BEV rasters from PointNet blocks"
        )
    )
    config = config_parser.load()

    BLOCK_ROOT = Path(config.model_data_path)
    OUT_ROOT   = Path(str(config.model_data_path) + "_BEV")

    for split in ["train", "test"]:

        in_dir  = BLOCK_ROOT / split
        out_dir = OUT_ROOT / split
        out_dir.mkdir(parents=True, exist_ok=True)

        files = glob.glob(str(in_dir / "*.npz"))
        print(f"{split}: {len(files)} blocks")

        for fpath in files:

            data = np.load(fpath)
            pts = data["points"]

            density, z_max, z_mean, z_range = rasterize(pts, H, W)

            base = Path(fpath).stem
            out_path = out_dir / f"{base}_bev_{H}x{W}.npz"

            np.savez_compressed(
                out_path,
                density=density,
                z_max=z_max,
                z_mean=z_mean,
                z_range=z_range
            )

    print("BEV generation complete.")


if __name__ == "__main__":
    main()
