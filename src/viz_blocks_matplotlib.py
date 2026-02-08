import os
import glob
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.config_parser import ConfigParser
from pathlib import Path


config_parser = ConfigParser(
    default_config_path="config/default.yaml",
    parser=argparse.ArgumentParser(description='3D Semantic Segmentation on DALES Dataset')
)
config = config_parser.load()
BLOCK_DIR = Path(config.model_data_path) / "train"
N_BLOCKS_TO_VIEW = config.viz_2d['n_blocks_to_view']
MAX_POINTS_TO_SHOW = config.viz_2d['max_points_to_view']
COLOR_MAP = config.viz_2d['color_mapping']


def main():
    files = glob.glob(os.path.join(BLOCK_DIR, "*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files found in: {BLOCK_DIR}")

    picks = random.sample(files, k=min(N_BLOCKS_TO_VIEW, len(files)))

    for i, fpath in enumerate(picks, 1):
        d = np.load(fpath)
        pts = d["points"]
        lbl = d["labels"].astype(int)

        # Optional subsample (not usually needed since pts is 4096)
        if len(pts) > MAX_POINTS_TO_SHOW:
            idx = np.random.choice(len(pts), MAX_POINTS_TO_SHOW, replace=False)
            pts, lbl = pts[idx], lbl[idx]

        fname = os.path.basename(fpath)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        for cls in sorted(set(lbl)):
            m = lbl == cls
            ax.scatter(
                pts[m, 0], pts[m, 1], pts[m, 2],
                s=2, c=COLOR_MAP.get(cls, "black"),
                label=str(cls)
            )

        ax.set_title(f"Block {i}: {fname}")
        ax.legend(markerscale=4, fontsize=8, loc="upper right")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
