import argparse
from pathlib import Path

import laspy
import numpy as np
from plyfile import PlyData


INTENSITY_NAMES = ["intensity", "Intensity", "intensities"]


def read_ply_intensity(ply_path: Path) -> np.ndarray:
    plydata = PlyData.read(str(ply_path))
    # Element may be named "vertex", "testing", or something else — use the first one
    vertex = plydata.elements[0]
    for name in INTENSITY_NAMES:
        if name in vertex.data.dtype.names:
            arr = np.asarray(vertex[name], dtype=np.float64)
            # scale [0,1] floats to uint16
            if arr.max() <= 1.0:
                arr = arr * 65535.0
            return np.clip(arr, 0, 65535).astype(np.uint16)
    raise RuntimeError(f"No intensity attribute in {ply_path}. Available: {vertex.data.dtype.names}")


def resolve_ply_path(ply_split: Path, rel: Path) -> Path:
    stem = ply_split / rel.with_suffix("")
    preferred = stem.with_name(stem.name + "_new").with_suffix(".ply")
    fallback = stem.with_suffix(".ply")
    return preferred if preferred.exists() else fallback


def process_split(las_root: Path, ply_root: Path, out_root: Path, split: str):
    las_split = las_root / split
    ply_split = ply_root / split
    out_split = out_root / split

    las_files = sorted(las_split.rglob("*.las"))
    print(f"\n=== {split.upper()} | {len(las_files)} files ===")

    ok = failed = 0
    for las_path in las_files:
        rel = las_path.relative_to(las_split)
        ply_path = resolve_ply_path(ply_split, rel)
        out_path = out_split / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not ply_path.exists():
            print(f"[FAIL] No PLY for {las_path.name}")
            failed += 1
            continue

        try:
            las = laspy.read(str(las_path))
            intensity = read_ply_intensity(ply_path)
            if len(las.points) != len(intensity):
                raise RuntimeError(f"Point count mismatch: LAS={len(las.points)}, PLY={len(intensity)}")
            las.intensity = intensity
            las.write(str(out_path))
            print(f"[OK]   {las_path.name}")
            ok += 1
        except Exception as exc:
            print(f"[FAIL] {las_path.name} | {exc}")
            failed += 1

    print(f"Summary {split}: ok={ok}, failed={failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy intensity from PLY files into LAS files"
    )
    parser.add_argument("--las-root", type=Path, default=Path("data") / "dales_las")
    parser.add_argument("--ply-root", type=Path, default=Path("data") / "dales_ply")
    parser.add_argument("--out-root", type=Path, default=Path("data") / "dales_las_ext")
    args = parser.parse_args()

    for split in ("train", "test"):
        process_split(args.las_root, args.ply_root, args.out_root, split)


if __name__ == "__main__":
    main()