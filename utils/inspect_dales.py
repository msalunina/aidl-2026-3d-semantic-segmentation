import os
import glob
import laspy
import numpy as np

print(">>> inspect_dales.py started")

LAS_DIR = r"E:\Dales\dales_semantic_segmentation_las\dales_las\train"
print("Checking directory:", LAS_DIR)
print("Exists:", os.path.exists(LAS_DIR))

files = glob.glob(os.path.join(LAS_DIR, "*.las"))
print("Found LAS files:", len(files))

if len(files) == 0:
    raise RuntimeError("No .las files found – check path")

las_path = files[0]
print("Reading:", las_path)

las = laspy.read(las_path)

xyz = np.vstack((las.x, las.y, las.z)).T
labels = las.classification

print("Points:", xyz.shape)
print("Unique labels:", np.unique(labels))
