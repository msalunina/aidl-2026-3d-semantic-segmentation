import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

# PointNet blocks output folder
BLOCK_DIR = r"E:\Dales\pointnet_blocks\train"  # or r"E:\Dales\pointnet_blocks\test"
N_BLOCKS_TO_VIEW = 3
MAX_POINTS_TO_SHOW = 4096  # blocks are already 4096, but keep for safety

# Your 5-class mapping:
# 0 Ground, 1 Vegetation, 2 Building, 3 Vehicle, 4 Utility, -1 Ignore
COLOR_MAP = {
    -1: "gray",
     0: "blue",
     1: "green",
     2: "red",
     3: "gold",
     4: "orange",
}

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
