import laspy
import numpy as np
import os
from pathlib import Path

DALES_ROOT = Path("/Users/oortega/SACO/Cursos/UPC_postgrau_AIDL/TREBALL_FINAL/codi/aidl-project-main/data/dales_las")  # wherever the .las live
las_path = DALES_ROOT / "train/5180_54435.las"

las = laspy.read(las_path)

for name in dir(las):
    print(name)

print("\n(list(las.point_format.dimension_names)):")
dims = list(las.point_format.dimension_names)
print(dims)


las_files = sorted(DALES_ROOT.rglob("*.las"))            # Path.glob("*.las"): all files in this folder that match *.las

for file_path in las_files:     # file_path is not a string but a Path object and it has attribute .name
    las = laspy.read(file_path)
    intensity = np.unique(las.points["intensity"])
    print(file_path.name, intensity)





