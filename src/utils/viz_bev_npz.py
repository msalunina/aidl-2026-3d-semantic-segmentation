import os
import glob
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_bev_npz(path: str):
    d = np.load(path)
    density = d["density"]
    z_max = d["z_max"]
    z_mean = d["z_mean"]
    z_range = d["z_range"]
    return density, z_max, z_mean, z_range


def show_bev(density, z_max, z_mean, z_range, title="BEV"):
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(density, origin="lower")
    ax1.set_title("density = log(1 + count)")
    ax1.set_xticks([]); ax1.set_yticks([])

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(z_max, origin="lower")
    ax2.set_title("z_max")
    ax2.set_xticks([]); ax2.set_yticks([])

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(z_mean, origin="lower")
    ax3.set_title("z_mean")
    ax3.set_xticks([]); ax3.set_yticks([])

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(z_range, origin="lower")
    ax4.set_title("z_range = z_max - z_min")
    ax4.set_xticks([]); ax4.set_yticks([])

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize BEV .npz rasters")
    parser.add_argument("--bev_root", type=str, default="E:/Dales/pointnet_blocks_BEV",
                        help="Root folder containing train/test BEV folders")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                        help="Which split to visualize")
    parser.add_argument("--n", type=int, default=3, help="Number of BEV files to visualize")
    parser.add_argument("--pattern", type=str, default="*.npz", help="Glob pattern")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--stats", action="store_true", help="Print min/max stats per channel")
    args = parser.parse_args()

    random.seed(args.seed)

    bev_dir = Path(args.bev_root) / args.split
    files = sorted(glob.glob(str(bev_dir / args.pattern)))

    if not files:
        raise RuntimeError(f"No BEV .npz files found in {bev_dir} with pattern {args.pattern}")

    picks = random.sample(files, k=min(args.n, len(files)))

    for i, fpath in enumerate(picks, 1):
        density, z_max, z_mean, z_range = load_bev_npz(fpath)
        fname = os.path.basename(fpath)

        if args.stats:
            print(f"\n[{i}] {fname}")
            print(f"  density: min={density.min():.3f} max={density.max():.3f}")
            print(f"  z_max  : min={z_max.min():.3f} max={z_max.max():.3f}")
            print(f"  z_mean : min={z_mean.min():.3f} max={z_mean.max():.3f}")
            print(f"  z_range: min={z_range.min():.3f} max={z_range.max():.3f}")

        show_bev(density, z_max, z_mean, z_range, title=f"BEV {i}: {fname}")


if __name__ == "__main__":
    main()
