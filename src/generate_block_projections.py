import os
import glob
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---- Make imports work when running this file directly ----
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config_parser import ConfigParser


def save_projection_png(
    pts: np.ndarray,
    lbl: np.ndarray,
    out_png: Path,
    plane: str,
    color_map: dict,
    dpi: int,
    point_size: float,
    alpha: float,
):
    """Save a projection scatter plot as a PNG. plane: 'xy' or 'xz'."""
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if plane == "xy":
        a, b = 0, 1
    elif plane == "xz":
        a, b = 0, 2
    else:
        raise ValueError("plane must be 'xy' or 'xz'")

    fig = plt.figure(figsize=(4, 4), dpi=dpi)
    ax = fig.add_subplot(111)

    for cls in sorted(set(lbl.tolist())):
        m = lbl == cls
        ax.scatter(
            pts[m, a], pts[m, b],
            s=point_size,
            c=color_map.get(int(cls), "black"),
            alpha=alpha,
            linewidths=0
        )

    # Blocks are normalized -> consistent bounds
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")

    # Image-like output
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout(pad=0.1)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate XY/XZ projection images for PointNet blocks (train+test).")
    parser.add_argument("--max_blocks", type=int, default=None,
                        help="Limit blocks per split (debug).")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--point_size", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.85)
    args = parser.parse_args()

    config_parser = ConfigParser(
        default_config_path="config/default.yaml",
        parser=argparse.ArgumentParser(add_help=False)
    )
    config = config_parser.load()

    blocks_root = Path(config.model_data_path)  # e.g. E:/Dales/pointnet_blocks

    # Output dirs are siblings of pointnet_blocks
    xy_root = blocks_root.parent / "pointnet_blocks_XY_projections"
    xz_root = blocks_root.parent / "pointnet_blocks_XZ_projections"

    # Reuse our YAML colors
    color_map = config.viz_2d["color_mapping"]

    for split in ["train", "test"]:
        in_dir = blocks_root / split
        if not in_dir.exists():
            raise FileNotFoundError(f"Blocks folder not found: {in_dir}")

        files = sorted(glob.glob(str(in_dir / "*.npz")))
        if args.max_blocks is not None:
            files = files[: args.max_blocks]

        out_xy = xy_root / split
        out_xz = xz_root / split
        out_xy.mkdir(parents=True, exist_ok=True)
        out_xz.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {split.upper()} ===")
        print("Input blocks:", in_dir)
        print("Blocks found:", len(files))
        print("XY out:", out_xy)
        print("XZ out:", out_xz)

        for k, fpath in enumerate(files, 1):
            d = np.load(fpath)
            pts = d["points"].astype(np.float32)
            lbl = d["labels"].astype(int)

            stem = Path(fpath).stem
            out_xy_png = out_xy / f"{stem}.png"
            out_xz_png = out_xz / f"{stem}.png"

            save_projection_png(pts, lbl, out_xy_png, "xy", color_map, args.dpi, args.point_size, args.alpha)
            save_projection_png(pts, lbl, out_xz_png, "xz", color_map, args.dpi, args.point_size, args.alpha)

            if k % 500 == 0:
                print(f"  processed {k}/{len(files)}")

        print(f"Done {split} ✅")


if __name__ == "__main__":
    main()
