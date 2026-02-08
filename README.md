# aidl-2026-3d-semantic-segmentation

## Folder Structure

```
aidl-2026-3d-semantic-segmentation/
├── config/                       # Configuration files
│   └── default.yaml              # Default configuration (paths, training, preprocessing, viz)
├── data/                         # DALES dataset
│   ├── dales_las/                # Raw LAS point cloud files
│   │   ├── train/                # Training tiles
│   │   └── test/                 # Test tiles
│   └── dales_blocks/             # Preprocessed blocks (NPZ format)
│   │   ├── train/                # Training blocks
│   │   └── test/                 # Test blocks
├── figs/                         # Figures and visualizations
├── logs/                         # Training logs
├── model_objects/                # Saved model checkpoints
│   └── pointnet.pth
├── src/                          # Source code
│   ├── main.py                   # Main training script
│   ├── compute_class_frequencies.py  # Analyze class distribution
│   ├── convert_las_to_blocks.py  # Preprocess LAS files into blocks
│   ├── viz_blocks_matplotlib.py  # 2D visualization script
│   ├── viz_blocks_open3d.py      # 3D visualization script
│   ├── models/                   # Model architectures
│   │   └── pointnet.py           # PointNet implementation
│   └── utils/                    # Utility modules
│       ├── config_parser.py      # Configuration loading and parsing
│       ├── dales_label_map.py    # DALES label definitions
│       ├── dataset.py            # Dataset loaders
│       ├── evaluator.py          # Evaluation loop
│       └── trainer.py            # Training loop
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Quick start

Environment:
```bash
conda create -n aidl-2026-project python=3.10 -y
conda activate aidl-2026-project
pip install -r requirements.txt
```

## Configuration

The project uses a hybrid configuration system that balances flexibility and simplicity:

### Primary Configuration: YAML File

All project settings are defined in `config/default.yaml`, which includes:

**Paths:**
- Data directories (raw LAS files, preprocessed blocks)
- Output directories (logs, figures, saved models)

**Data Preprocessing:**
- Block size, stride, and overlap settings
- Point sampling parameters (e.g., 4096 points per block)
- Class mapping (original DALES labels → simplified classes)
- Minimum points threshold for valid blocks

**Training Hyperparameters:**
- Model architecture selection
- Number of input channels and output classes
- Points per training sample, batch size, epochs
- Learning rate, dropout rate, optimizer choice

**Visualization:**
- 2D and 3D visualization settings
- Number of blocks to display
- Color mappings for each class

To modify any of these settings, edit `config/default.yaml` directly.

### Command-Line Override: Training Parameters Only

For quick experimentation, you can override **training hyperparameters** from the command line without editing the YAML file:

```bash
python src/main.py --num_epochs 100 --batch_size 16 --learning_rate 0.0005 --dropout_rate 0.5
```

**Available command-line arguments:**
- `--model_name`: Model architecture (choices: pointnet)
- `--num_channels`: Number of input channels (e.g., 3 for XYZ)
- `--num_points`: Number of points per training sample
- `--batch_size`: Batch size for training
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate for optimizer
- `--dropout_rate`: Dropout rate for regularization
- `--optimizer`: Optimizer to use (choices: adam)

**Note:** Paths, data preprocessing settings, and visualization parameters cannot be overridden from the command line and must be configured in the YAML file.