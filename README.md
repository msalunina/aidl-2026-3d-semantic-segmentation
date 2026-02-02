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

## Configuration

The project supports two ways to configure settings:

### Option 1: Modify default configuration file
Edit the `config/default.yaml` file to change the default values, or create your own configuration file based on this template and update the config path in `main.py`.

### Option 2: Pass arguments via command line
Override any default setting by passing command-line arguments:

```bash
python src/main.py --num_epochs 100 --batch_size 16 --learning_rate 0.0005 --dropout_rate 0.5 --num_points 2048
```

**Available command-line arguments:**

**Paths:**
- `--data_path`: Path to data directory
- `--logs_path`: Path to logs directory
- `--figures_path`: Path to figures directory
- `--models_path`: Path to saved models directory

**Model:**
- `--model_name`: Model architecture (choices: pointnet)
- `--num_classes`: Number of output classes
- `--num_channels`: Number of input channels

**Training:**
- `--num_points`: Number of points per sample
- `--batch_size`: Batch size for training
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate for optimizer
- `--dropout_rate`: Dropout rate for model
- `--optimizer`: Optimizer to use (choices: adam)

**Train/Test Split:**
- `--train_ratio`: Training set ratio
- `--test_ratio`: Test set ratio