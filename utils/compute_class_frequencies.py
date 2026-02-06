import glob
import numpy as np
from collections import Counter

# Path to our PointNet-ready blocks
BLOCK_DIR = r"E:\Dales\pointnet_blocks\train"

# Classes:
# 0 = Ground
# 1 = Vegetation
# 2 = Building
# 3 = Vehicle
# 4 = Utility
IGNORE_LABEL = -1

def main():
    files = glob.glob(f"{BLOCK_DIR}\\*.npz")
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
    for cls in sorted(counter.keys()):
        count = counter[cls]
        pct = 100.0 * count / total_points
        print(f"Class {cls}: {count:,} points ({pct:.3f}%)")

    print(f"\nTotal labeled points: {total_points:,}")

    # ---- Compute weights ----
    # Inverse frequency (simple, effective baseline)
    freqs = np.array([counter.get(i, 0) for i in range(5)], dtype=np.float64)
    freqs = freqs / freqs.sum()

    weights = 1.0 / (freqs + 1e-12)
    weights = weights / weights.mean()  # normalize for stability

    print("\n=== Suggested loss weights ===")
    for i, w in enumerate(weights):
        print(f"Class {i}: weight = {w:.2f}")

    print("\nPyTorch tensor:")
    print(f"torch.tensor({weights.tolist()}, dtype=torch.float32)")

if __name__ == "__main__":
    main()
