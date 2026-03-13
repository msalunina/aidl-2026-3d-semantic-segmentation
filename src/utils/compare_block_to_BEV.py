"""
Compare point cloud block to BEV

Run example:
    python .\compare_block_to_BEV.py --split train

Or with reproducible random choice:
    python .\compare_block_to_BEV.py --split train --seed 42
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def find_project_root(start: Path) -> Path:
    """Walk upward until we find the project root containing /data and /config."""
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "data").exists() and (p / "config").exists():
            return p
    raise FileNotFoundError("Could not find project root containing 'data' and 'config' folders.")


def load_yaml_color_mapping(project_root: Path) -> dict:
    """
    Read visualization.2d.color_mapping from config/default.yaml
    and return a dict with integer keys.
    """
    yaml_path = project_root / "config" / "default.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML config not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    color_mapping = cfg["visualization"]["2d"]["color_mapping"]
    return {int(k): v for k, v in color_mapping.items()}


def extract_tile_prefix(block_path: Path) -> str:
    """
    For a block like:
        5190_54435_b00236.npz
    return:
        5190_54435
    """
    stem = block_path.stem
    if "_b" in stem:
        return stem.split("_b")[0]
    return stem


def _read_npz_string_field(npz_obj, key: str):
    """
    Safely read a string-like field from npz saved as np.array([value]).
    Returns None if field is missing.
    """
    if key not in npz_obj.files:
        return None

    value = npz_obj[key]

    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size > 0:
            return str(value.flat[0])

    return str(value)


def find_matching_bev_path_from_metadata(block_npz, block_path: Path, project_root: Path) -> Path | None:
    """
    Use deterministic metadata stored inside the block file if available.
    Priority:
      1) bev_filename
      2) bev_key (+ suffix)
    """
    split = block_path.parent.name
    bev_dir = project_root / "data" / "dales_las_BEV_FULL" / split

    if not bev_dir.exists():
        raise FileNotFoundError(f"BEV directory not found: {bev_dir}")

    bev_filename = _read_npz_string_field(block_npz, "bev_filename")
    if bev_filename is not None:
        candidate = bev_dir / bev_filename
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Block contains bev_filename='{bev_filename}', but file does not exist:\n{candidate}"
        )

    bev_key = _read_npz_string_field(block_npz, "bev_key")
    if bev_key is not None:
        candidate = bev_dir / f"{bev_key}__bev_full_256x256.npz"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Block contains bev_key='{bev_key}', but expected file does not exist:\n{candidate}"
        )

    return None


def find_matching_bev_path_fallback(block_path: Path, project_root: Path, rng: random.Random | None = None) -> Path:
    """
    Fallback logic for older block files without metadata.

    Tries:
    1) old naming convention:
         <stem>_bev_256x256.npz
    2) exact stem with full-density suffix:
         <stem>__bev_full_256x256.npz
    3) tile-prefix wildcard match:
         <tile_prefix>*.npz

    If multiple candidates are found for the same tile prefix, pick one randomly.
    """
    split = block_path.parent.name
    bev_dir = project_root / "data" / "dales_las_BEV_FULL" / split
    if not bev_dir.exists():
        raise FileNotFoundError(f"BEV directory not found: {bev_dir}")

    stem = block_path.stem
    tile_prefix = extract_tile_prefix(block_path)

    candidate_1 = bev_dir / f"{stem}_bev_256x256.npz"
    if candidate_1.exists():
        return candidate_1

    candidate_2 = bev_dir / f"{stem}__bev_full_256x256.npz"
    if candidate_2.exists():
        return candidate_2

    candidates = sorted(bev_dir.glob(f"{tile_prefix}*.npz"))
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No BEV match found for block {block_path.name}.\n"
            f"Tried metadata, exact names, and wildcard prefix '{tile_prefix}*.npz' in {bev_dir}"
        )

    if len(candidates) > 1:
        chooser = rng if rng is not None else random
        chosen = chooser.choice(candidates)
        print(f"Found {len(candidates)} fallback BEV candidates for tile prefix '{tile_prefix}'.")
        print(f"Using: {chosen.name}")
        return chosen

    return candidates[0]


def pick_random_block(project_root: Path, split: str = "train", rng: random.Random | None = None) -> Path:
    block_dir = project_root / "data" / "dales_blocks" / split
    block_files = sorted(block_dir.glob("*.npz"))

    if len(block_files) == 0:
        raise FileNotFoundError(f"No block files found in {block_dir}")

    chooser = rng if rng is not None else random
    return chooser.choice(block_files)


def load_pair(block_path: Path, project_root: Path, rng: random.Random | None = None):
    if not block_path.exists():
        raise FileNotFoundError(f"Block file not found: {block_path}")

    block = np.load(block_path, allow_pickle=True)

    bev_path = find_matching_bev_path_from_metadata(block, block_path, project_root)
    if bev_path is None:
        bev_path = find_matching_bev_path_fallback(block_path, project_root, rng=rng)

    bev = np.load(bev_path, allow_pickle=True)

    return bev_path, block, bev


def labels_to_colors(labels: np.ndarray, color_mapping: dict):
    """Convert integer labels to matplotlib-compatible color names from YAML."""
    return [color_mapping.get(int(lbl), "black") for lbl in labels]


def visualize_random_block_and_bev(split: str = "train", seed: int | None = None):
    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)

    rng = random.Random(seed) if seed is not None else random
    color_mapping = load_yaml_color_mapping(project_root)

    block_path = pick_random_block(project_root, split=split, rng=rng)
    bev_path, block, bev = load_pair(block_path, project_root, rng=rng)

    points = block["points"]
    labels = block["labels"]
    xyz = points[:, :3]

    print("Block file:", block_path)
    print("BEV file  :", bev_path)
    print("Block keys:", block.files)
    print("BEV keys  :", bev.files)
    print("Points shape:", points.shape)
    print("Labels shape:", labels.shape)

    if "bev_key" in block.files:
        print("Block bev_key:", _read_npz_string_field(block, "bev_key"))
    if "bev_filename" in block.files:
        print("Block bev_filename:", _read_npz_string_field(block, "bev_filename"))

    print("YAML 2D color mapping:", color_mapping)

    point_colors = labels_to_colors(labels, color_mapping)
    available_bev_channels = [ch for ch in ["density", "z_max", "z_mean", "z_range"] if ch in bev.files]

    fig, axes = plt.subplots(
        1,
        1 + len(available_bev_channels),
        figsize=(5 * (1 + len(available_bev_channels)), 5)
    )

    if len(available_bev_channels) == 0:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = [axes]

    # XY scatter colored by label
    axes[0].scatter(xyz[:, 0], xyz[:, 1], c=point_colors, s=1)
    axes[0].set_title(f"Point cloud XY\n{block_path.name}")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].set_aspect("equal")

    # BEV channels
    for ax, ch in zip(axes[1:], available_bev_channels):
        ax.imshow(bev[ch], origin="lower")
        ax.set_title(ch)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pick a random PointNet block and show its matching BEV file."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "val"],
        help="Dataset split to sample from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling",
    )
    args = parser.parse_args()

    visualize_random_block_and_bev(split=args.split, seed=args.seed)