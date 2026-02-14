"""
inspect_las_file.py

Utility script to inspect raw DALES .las files.
Prints metadata, class distribution and sample points.
Used for dataset sanity checking.
"""
import os
from pathlib import Path

import laspy
import pandas as pd
import numpy as np
import yaml


def main():
    # Load config
    config_path = Path("config/default.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_data = cfg["paths"]["raw_data"]  # e.g. "E:/Dales/.../dales_las"
    las_path = Path(raw_data) / "train/5080_54435.las"

    print("LAS path:", las_path)
    if not las_path.exists():
        raise FileNotFoundError(f"Could not find: {las_path}")

    las = laspy.read(str(las_path))

    print("Number of points:", len(las.x))
    print("Available dimensions:", list(las.point_format.dimension_names))

    # Preview first 15 points (table-like)
    df = pd.DataFrame({
        "x": np.array(las.x[:15]),
        "y": np.array(las.y[:15]),
        "z": np.array(las.z[:15]),
        "intensity": np.array(las.intensity[:15]),
        "classification": np.array(las.classification[:15]),
        "return_number": np.array(las.return_number[:15]),
        "number_of_returns": np.array(las.number_of_returns[:15]),
        "scan_angle_rank": np.array(las.scan_angle_rank[:15]) if "scan_angle_rank" in las.point_format.dimension_names else None,
        "gps_time": np.array(las.gps_time[:15]) if "gps_time" in las.point_format.dimension_names else None,
    })

    print("\nFirst 15 rows:")
    print(df)


if __name__ == "__main__":
    main()
