import glob
import random
import argparse
import numpy as np
import open3d as o3d
import os
from utils.config_parser import ConfigParser
from pathlib import Path


config_parser = ConfigParser(
    default_config_path="config/default.yaml",
    parser=argparse.ArgumentParser(description='3D Semantic Segmentation on DALES Dataset')
)
config = config_parser.load()
BLOCK_DIR = Path(config.model_data_path) / "train"
N_BLOCKS_TO_VIEW = config.viz_3d['n_blocks_to_view']
MAX_POINTS_TO_SHOW = config.viz_3d['max_points_to_view']
COLOR_MAP = config.viz_3d['color_mapping']


def load_npz(path):
    d = np.load(path)
    pts = d["points"][:, :3]  # (4096,3)
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
    files = glob.glob(os.path.join(BLOCK_DIR, "*.npz"))
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
