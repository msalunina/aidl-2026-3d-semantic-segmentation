import glob
import random
import numpy as np
import open3d as o3d

# PointNet blocks output folder
BLOCK_DIR = r"E:\Dales\pointnet_blocks\train"  # or ...\test
N_BLOCKS_TO_VIEW = 5
MAX_POINTS_TO_SHOW = 20000  # render faster; blocks are 4096 anyway, so fine

# Your 5-class mapping:
# 0 Ground, 1 Vegetation, 2 Building, 3 Vehicle, 4 Utility, -1 Ignore
COLOR_MAP = {
    -1: (0.3, 0.3, 0.3),  # gray (ignored/unknown)
     0: (0.0, 0.0, 1.0),  # blue ground
     1: (0.0, 0.6, 0.0),  # green vegetation
     2: (1.0, 0.0, 0.0),  # red building
     3: (1.0, 0.85, 0.0), # yellow vehicle
     4: (1.0, 0.5, 0.0),  # orange utility
}

def load_npz(path):
    d = np.load(path)
    pts = d["points"]  # (4096,3)
    lbl = d["labels"]  # (4096,)
    return pts, lbl

def make_pcd(pts, lbl):
    n = len(pts)
    if n > MAX_POINTS_TO_SHOW:
        idx = np.random.choice(n, MAX_POINTS_TO_SHOW, replace=False)
        pts, lbl = pts[idx], lbl[idx]

    colors = np.array([COLOR_MAP.get(int(c), (1.0, 1.0, 1.0)) for c in lbl], dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def main():
    files = glob.glob(BLOCK_DIR + r"\*.npz")
    if not files:
        raise RuntimeError(f"No .npz files found in: {BLOCK_DIR}")

    picks = random.sample(files, k=min(N_BLOCKS_TO_VIEW, len(files)))

    for i, f in enumerate(picks, 1):
        pts, lbl = load_npz(f)
        uniq, cnt = np.unique(lbl, return_counts=True)
        dist = dict(zip([int(u) for u in uniq], [int(c) for c in cnt]))

        print(f"\n[{i}/{len(picks)}] {f}")
        print("Label counts:", dist)

        pcd = make_pcd(pts, lbl)
        o3d.visualization.draw_geometries([pcd], window_name=f"DALES Block {i}")

if __name__ == "__main__":
    main()
