# aidl-2026-3d-semantic-segmentation

!!!!! pointnet.py contains one implementation that has to be verified yet

we could split to do:

    - dataset.py (should include clean up of the classes / concatenation of the classes)
    - trainer.py
    - evaluator.py
    - main.py
    - analyze_dataset.py

## Folder Structure (PROPOSAL TO BE DISCUSSED AND UPDATED)

```
aidl-2026-3d-semantic-segmentation/
├── data/                         # terlidar/dales dataset
│   ├── .../
│   └── .../
├── figs/                         # Figures and visualizations
├── logs/                         # Training logs (we can commit&push only final ones)
├── model_objects/                # Saved model checkpoints
│   ├── pointnet.pth
│   └── other_model.pth
├── src/                          # Source code
│   ├── dataset.py                # Dataset loaders
│   ├── evaluator.py              # Contains class for measuring on test
│   ├── main.py                   # Main training script
│   ├── trainer.py                # Contains class for training a model
│   ├── models/                   # Model architectures
│   │   └── pointnet.py           # PointNet implementation
│   └── utils/                    # Utility functions (visualizations, metrics, logger)
├── analyze_dataset.py            # Script to see class distribution, channels; visualizations
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Quick start (PLEASE CHECK IF WORKS FOR EVERYONE; torch version is with GPU compatibility)

Environment:
```bash
conda create -n aidl-2026-project python=3.10 -y
conda activate aidl-2026-project
pip install -r requirements.txt
```