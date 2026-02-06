"""
DALES label mapping utilities.

Original DALES LAS labels observed in the dataset: 0..8
0 = unknown (ignored)

Simplified 5-class scheme:
0 -> Ground
1 -> Vegetation
2 -> Building
3 -> Vehicle (cars + trucks)
4 -> Utility (power lines + poles + fences)
"""

import numpy as np

# DALES LAS labels (as in your files): 0..8
# 0 = unknown (ignore)
DALES_TO_SIMPLIFIED = {
    1: 0,  # ground
    2: 1,  # vegetation
    8: 2,  # buildings
    3: 3,  # cars
    4: 3,  # trucks
    5: 4,  # power lines
    6: 4,  # poles
    7: 4,  # fences
}

# Label used to ignore points during training / loss computation
IGNORE_LABEL = -1


def remap_labels(las_labels: np.ndarray) -> np.ndarray:
    """
    Remap DALES LAS labels (0..8) to simplified 5-class scheme.

    Parameters
    ----------
    las_labels : np.ndarray
        Array of original DALES LAS labels.

    Returns
    -------
    np.ndarray
        Array of remapped labels:
        {0..4} for valid classes, IGNORE_LABEL (-1) for unknown.
    """
    out = np.full(las_labels.shape, IGNORE_LABEL, dtype=np.int64)
    for src, dst in DALES_TO_SIMPLIFIED.items():
        out[las_labels == src] = dst
    return out


if __name__ == "__main__":
    # Simple self-test
    test_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    print("Input labels :", test_labels)
    print("Mapped labels:", remap_labels(test_labels))
    print("DALES_TO_SIMPLIFIED =", DALES_TO_SIMPLIFIED)
    print("IGNORE_LABEL =", IGNORE_LABEL)
