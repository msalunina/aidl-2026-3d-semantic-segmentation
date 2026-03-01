"""
generate_full_density_bev_rasters_from_las.py

Generate "FULL-DENSITY BEV (not PointNet input)" rasters directly from raw LAS,
without PointNet sampling (no 4096-point subsampling).

For each LAS file:
- slide a 50m x 50m window with 25m stride (configurable)
- rasterize all points in the window into HxW BEV channels:
    - density  : log(1 + count)
    - z_max    : max Z in cell
    - z_mean   : mean Z in cell
    - z_range  : z_max - z_min

Output:
  <las_root>_BEV_FULL/{train,test}/<las_stem>__x<ix>_y<iy>__bev_full_<H>x<W>.npz

Each output NPZ contains:
  density, z_max, z_mean, z_range, meta (dict-like via np.savez), note (string)
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import laspy


def print_bev_resolution(block_size_m: float, H: int, W: int):
    px_x = block_size_m / float(W)
    px_y = block_size_m / float(H)
    print("\n" + "=" * 15 + " BEV RESOLUTION " + "=" * 15)
    print(f"Block size         : {block_size_m:.2f} m x {block_size_m:.2f} m")
    print(f"Grid resolution    : {H} x {W}")
    print(f"Pixel size (X)     : {px_x:.4f} m/pixel")
    print(f"Pixel size (Y)     : {px_y:.4f} m/pixel")
    print("=" * 47 + "\n")


def rasterize_full_points_xy(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x0: float,
    y0: float,
    block_size_m: float,
    H: int,
    W: int,
):
    """
    Rasterize all points within one (x0,y0) .. (x0+block_size, y0+block_size) window.
    Assumes x,y are already filtered to this window.
    """
    # Map meters -> pixel indices
    # u corresponds to X axis (0..W-1), v corresponds to Y axis (0..H-1)
    u = ((x - x0) / block_size_m * (W - 1)).astype(np.int32)
    v = ((y - y0) / block_size_m * (H - 1)).astype(np.int32)
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)

    density = np.zeros((H, W), dtype=np.float32)
    z_max   = np.full((H, W), -np.inf, dtype=np.float32)
    z_min   = np.full((H, W), +np.inf, dtype=np.float32)
    z_sum   = np.zeros((H, W), dtype=np.float32)

    # Iterate points (simple + clear; fast enough for typical blocks)
    for i in range(z.shape[0]):
        ui, vi = u[i], v[i]
        density[vi, ui] += 1.0
        z_sum[vi, ui] += z[i]
        if z[i] > z_max[vi, ui]:
            z_max[vi, ui] = z[i]
        if z[i] < z_min[vi, ui]:
            z_min[vi, ui] = z[i]

    mask = density > 0
    z_mean = np.zeros((H, W), dtype=np.float32)
    z_mean[mask] = z_sum[mask] / density[mask]

    z_range = np.zeros((H, W), dtype=np.float32)
    z_range[mask] = z_max[mask] - z_min[mask]

    density = np.log1p(density)

    # Fill empty cells with 0 for visualization consistency
    z_max[~mask] = 0.0
    z_mean[~mask] = 0.0
    z_range[~mask] = 0.0

    return density, z_max, z_mean, z_range, int(mask.sum())


def main():
    parser = argparse.ArgumentParser(
        description="Generate FULL-DENSITY BEV rasters from LAS (not PointNet input)."
    )
    parser.add_argument(
        "--las_root",
        type=str,
        required=True,
        help=r"Root folder containing dales_las/{train,test} or directly train/test.",
    )
    parser.add_argument("--split", type=str, default=None, choices=[None, "train", "test"],
                        help="If provided, only process that split. If omitted, tries train and test.")
    parser.add_argument("--H", type=int, default=256, help="Raster height")
    parser.add_argument("--W", type=int, default=256, help="Raster width")
    parser.add_argument("--block_size_m", type=float, default=50.0, help="Block size in meters")
    parser.add_argument("--stride_m", type=float, default=25.0, help="Stride in meters")
    parser.add_argument("--min_points", type=int, default=200, help="Skip windows with fewer points than this")
    parser.add_argument("--pattern", type=str, default="*.las", help="LAS filename glob pattern")
    parser.add_argument("--out_root", type=str, default=None,
                        help="Optional output root. Default: <las_root>_BEV_FULL")
    args = parser.parse_args()

    las_root = Path(args.las_root)
    out_root = Path(args.out_root) if args.out_root else Path(str(las_root) + "_BEV_FULL")

    # Determine split folders
    splits_to_process = []
    if args.split:
        splits_to_process = [args.split]
    else:
        # Try typical structure: <las_root>/train and <las_root>/test
        # If not present, fall back to processing las_root itself as a single "train"
        if (las_root / "train").exists() or (las_root / "test").exists():
            if (las_root / "train").exists():
                splits_to_process.append("train")
            if (las_root / "test").exists():
                splits_to_process.append("test")
        else:
            splits_to_process = ["train"]  # logical label; input will be las_root

    print(f"Input LAS root : {las_root}")
    print(f"Output root    : {out_root}")
    print(f"Block size     : {args.block_size_m:.2f} m   Stride: {args.stride_m:.2f} m")
    print(f"Raster         : {args.H} x {args.W}")
    print_bev_resolution(args.block_size_m, args.H, args.W)
    print('NOTE: Outputs are labeled as "full-density BEV (not PointNet input)".\n')

    for split in splits_to_process:
        in_dir = (las_root / split) if (las_root / split).exists() else las_root
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        las_files = sorted(glob.glob(str(in_dir / args.pattern)))
        print(f"{split}: {len(las_files)} LAS files in {in_dir}")

        for las_path in las_files:
            las_path = Path(las_path)
            las = laspy.read(las_path)

            # Use scaled coordinates (las.x, las.y, las.z are float arrays)
            x = np.asarray(las.x, dtype=np.float64)
            y = np.asarray(las.y, dtype=np.float64)
            z = np.asarray(las.z, dtype=np.float32)

            x_min, x_max = float(x.min()), float(x.max())
            y_min, y_max = float(y.min()), float(y.max())

            bs = args.block_size_m
            st = args.stride_m

            # Tile origin grid
            xs = np.arange(x_min, x_max - bs + 1e-6, st, dtype=np.float64)
            ys = np.arange(y_min, y_max - bs + 1e-6, st, dtype=np.float64)

            stem = las_path.stem
            kept = 0
            total = 0

            for ix, x0 in enumerate(xs):
                x1 = x0 + bs
                # Pre-filter X to reduce work
                mx = (x >= x0) & (x < x1)
                if not np.any(mx):
                    continue

                for iy, y0 in enumerate(ys):
                    y1 = y0 + bs
                    m = mx & (y >= y0) & (y < y1)
                    if not np.any(m):
                        continue

                    total += 1
                    if int(m.sum()) < args.min_points:
                        continue

                    density, z_max, z_mean, z_range, nonempty = rasterize_full_points_xy(
                        x[m], y[m], z[m],
                        x0=x0, y0=y0,
                        block_size_m=bs,
                        H=args.H, W=args.W
                    )

                    out_path = out_dir / f"{stem}__x{ix:04d}_y{iy:04d}__bev_full_{args.H}x{args.W}.npz"

                    meta = {
                        "source_las": str(las_path),
                        "split": split,
                        "window_x0": float(x0),
                        "window_y0": float(y0),
                        "block_size_m": float(bs),
                        "stride_m": float(st),
                        "H": int(args.H),
                        "W": int(args.W),
                        "num_points_in_window": int(m.sum()),
                        "nonempty_pixels": int(nonempty),
                        "pixel_size_m_x": float(bs / args.W),
                        "pixel_size_m_y": float(bs / args.H),
                    }

                    np.savez_compressed(
                        out_path,
                        density=density,
                        z_max=z_max,
                        z_mean=z_mean,
                        z_range=z_range,
                        meta=meta,
                        note="full-density BEV (not PointNet input)",
                    )
                    kept += 1

            print(f"  {las_path.name}: windows kept {kept} / candidates {total}")

    print("\nFULL-DENSITY BEV generation complete.")


if __name__ == "__main__":
    main()
