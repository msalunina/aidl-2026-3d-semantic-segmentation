import glob
import argparse
import numpy as np
from collections import Counter
from utils.config_parser import ConfigParser
from pathlib import Path


config_parser = ConfigParser(
    default_config_path="config/default.yaml",
    parser=argparse.ArgumentParser(description='3D Semantic Segmentation on DALES Dataset')
)
config = config_parser.load()
BLOCK_DIR = Path(config.model_data_path) / "train"
IGNORE_LABEL = config.ignore_label


def compute_focal_weights(counter, total_points, method='sqrt_inv_freq'):
    """
    Compute class weights optimized for Focal Loss.
    
    Methods:
    - 'sqrt_inv_freq': Square root of inverse frequency (recommended for Focal Loss)
    - 'moderate': Moderate inverse frequency with clipping
    """
    freqs = np.array([counter.get(i, 0) for i in range(5)], dtype=np.float64)
    freqs = freqs / freqs.sum()
    
    if method == 'sqrt_inv_freq':
        # Square root of inverse frequency - less aggressive
        weights = np.sqrt(1.0 / (freqs + 1e-6))
        weights = weights / weights.mean()
        
    elif method == 'moderate':
        # Inverse frequency with clipping
        weights = 1.0 / (freqs + 1e-6)
        weights = weights / weights.mean()
        # Clip extreme weights to avoid instability
        weights = np.clip(weights, 0.5, 5.0)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return weights


def main():
    files = list(Path(BLOCK_DIR).glob("*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files found in {BLOCK_DIR}")

    print(f"Found {len(files)} training blocks")

    counter = Counter()
    total_points = 0

    for f in files:
        data = np.load(f)
        labels = data["labels"]

        # Ignore unknown label
        labels = labels[labels != IGNORE_LABEL]

        total_points += labels.size
        counter.update(labels.tolist())

    print("\n=== Class frequencies ===")
    class_names = ["Ground", "Vegetation", "Buildings", "Vehicle", "Utility"]
    for cls in sorted(counter.keys()):
        count = counter[cls]
        pct = 100.0 * count / total_points
        print(f"Class {cls} ({class_names[cls]:11s}): {count:,} points ({pct:.3f}%)")

    print(f"\nTotal labeled points: {total_points:,}")

    # ---- Compute weights ----
    # Compute different weight schemes
    print("=" * 70)
    print("RECOMMENDED WEIGHTS FOR FOCAL LOSS (gamma=2.0)")
    print("=" * 70)
    
    methods = [
        ('sqrt_inv_freq', 'Square Root Inv Freq (Recommended)'),
        ('moderate', 'Moderate Inv Freq (Clipped)')
    ]
    
    for method_name, method_label in methods:
        weights = compute_focal_weights(counter, total_points, method=method_name)
        
        print(f"\n{method_label}:")
        print("-" * 70)
        for i, w in enumerate(weights):
            print(f"  Class {i} ({class_names[i]:11s}): weight = {w:.4f}")
        
        print(f"\n  Config YAML format:")
        print(f"  loss_weights:")
        for w in weights:
            print(f"    - {w}")
    
    # Also show uniform weights as baseline
    print("\n" + "=" * 70)
    print("BASELINE: Uniform Weights (Let Focal Loss handle everything)")
    print("=" * 70)
    print("loss_weights:")
    for i in range(5):
        print(f"  - 1.0")

if __name__ == "__main__":
    main()
