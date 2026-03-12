# 3D Semantic Segmentation

## Table of Contents

- [3D Semantic Segmentation](#3d-semantic-segmentation)
  - [Table of Contents](#table-of-contents)
  - [Folder Structure](#folder-structure)
  - [How to Run the Code](#how-to-run-the-code)
    - [Local Setup](#local-setup)
    - [DALES Segmentation](#dales-segmentation)
      - [Preprocessing](#preprocessing)
      - [Training](#training)
      - [Testing](#testing)
      - [Visualization](#visualization)
    - [Running on Google Cloud VM](#running-on-google-cloud-vm)
      - [Environment setup](#environment-setup)
      - [NVIDIA driver installation](#nvidia-driver-installation)
      - [Verify GPU access](#verify-gpu-access)
  - [Experiment Tracking (W\&B)](#experiment-tracking-wb)
    - [What is logged](#what-is-logged)
    - [Configuration](#configuration)
    - [Offline usage](#offline-usage)
  - [PointNet Architecture](#pointnet-architecture)
    - [Why PointNet for aerial LiDAR?](#why-pointnet-for-aerial-lidar)
    - [Implementation (`src/models/pointnet.py`)](#implementation-srcmodelspointnetpy)
  - [Data Augmentation](#data-augmentation)
    - [Hypothesis](#hypothesis)
    - [Implementation (`src/models/dataset.py`)](#implementation-srcmodelsdatasetpy)
    - [Experiment Setup (TO BE ADDED)](#experiment-setup-to-be-added)
    - [Results (TO BE ADDED)](#results-to-be-added)
    - [Conclusions (TO BE REVISED)](#conclusions-to-be-revised)
  - [Focal Loss](#focal-loss)
    - [Hypothesis](#hypothesis-1)
    - [Implementation (`src/utils/focal_loss.py`)](#implementation-srcutilsfocal_losspy)
    - [Experiment Setup (TO BE ADDED)](#experiment-setup-to-be-added-1)
    - [Results (TO BE ADDED)](#results-to-be-added-1)
    - [Conclusions (TO BE ADDED)](#conclusions-to-be-added)
  - [Optimizer and Learning Rate](#optimizer-and-learning-rate)
    - [Configuration](#configuration-1)
  - [Class Weight Strategies](#class-weight-strategies)
    - [Hypothesis](#hypothesis-2)
    - [Experiment Setup](#experiment-setup)
    - [Results](#results)
    - [Conclusions](#conclusions)
  - [Future Work](#future-work)

---

## Folder Structure

```
aidl-2026-3d-semantic-segmentation/
├── config/
│   └── default.yaml                 # All hyperparameters, paths, and preprocessing settings
├── data/
│   ├── dales_las/                   # Raw LAS point cloud tiles
│   │   ├── train/
│   │   └── test/
│   ├── dales_las_BEV_FULL/          # BEV rasters for blocks
│   │   ├── train/
│   │   └── test/
│   └── dales_blocks/                # Preprocessed NPZ blocks
│       ├── train/
│       └── test/
├── figs/                            # Saved figures and visualizations
├── logs/                            # Training logs
├── model_objects/                   # Saved final model (.pth / .pt)
├── snapshots/                       # Per-epoch snapshots (snap_interval in config)
├── src/
│   ├── main.py                      # DALES training entry point
│   ├── test.py                      # DALES evaluation entry point
│   ├── train_shapenet.py            # ShapeNet training entry point
│   ├── convert_las_to_blocks.py     # Preprocessing: LAS → NPZ blocks
│   ├── compute_class_frequencies.py # Compute class frequencies & corresponding weights
│   ├── tune_ray.py                  # Ray Tune hyperparameter search
│   ├── viz_blocks_matplotlib.py     # 2D block visualization
│   ├── viz_blocks_open3d.py         # 3D Open3D visualization
│   ├── viz_critical_points.py       # PointNet critical point visualization
│   ├── viz_pred_vs_label.py         # Prediction vs. ground truth comparison
│   ├── models/
│   │   ├── pointnet.py              # PointNet segmentation model
│   │   ├── pointnetplusplus.py      # PointNet++ (baseline comparison)
│   │   └── img_encoder.py           # 2D CNN encoder for BEV features
│   └── utils/
│       ├── config_parser.py
│       ├── dales_label_map.py
│       ├── dataset.py               # DALES Dataset loader + augmentation
│       ├── shapenet_dataset.py      # ShapeNet Dataset loader + augmentation
│       ├── evaluator.py
│       ├── focal_loss.py            # Focal Loss implementation
│       ├── trainer.py               # Training loop + W&B logging
│       └── trainer_for_ray.py       # Trainer adapted for Ray Tune
├── requirements.txt
└── wandb/                           # Offline W&B run cache
```

---

## How to Run the Code

### Local Setup

```bash
conda create -n aidl-2026-project python=3.10 -y
conda activate aidl-2026-project
pip install --no-cache-dir -r requirements.txt
wandb login   # one-time setup, stores API key locally
```

### DALES Segmentation 

#### Preprocessing

Download DALES las files to `./data/dales_las`. Data can be requested using the form provided at the following [page](https://sites.google.com/a/udayton.edu/vasari1/research/earth-vision/dales).

Create BEV rasters by running (run from the repo root):

```bash
python src/generate_full_density_bev_rasters_from_las.py --las_root "./data/dales_las"
```

Convert raw LAS tiles to fixed-size NPZ blocks (run from the repo root):

```bash
python src/convert_las_to_blocks.py
```

Block size, stride, and feature extraction settings are configured in `config/default.yaml` under `data_preprocessing:`.

#### Training

```bash
python src/main.py
```

Override training hyperparameters without editing the YAML:

```bash
python src/main.py --num_epochs 100 --batch_size 16 --learning_rate 0.001 --dropout_rate 0.3
```

Available CLI arguments: `--model_name`, `--num_channels`, `--num_points`, `--batch_size`, `--num_epochs`, `--learning_rate`, `--dropout_rate`, `--optimizer`.

Paths, preprocessing settings, and loss weights must be changed in `config/default.yaml` directly.

#### Testing

```bash
python src/test.py
```

Results (accuracy, mIoU, per-class IoU) are printed to stdout and logged to W&B.

#### Visualization

```bash
python src/viz_blocks_matplotlib.py    # 2D bird's-eye view of blocks
python src/viz_blocks_open3d.py        # Interactive 3D point cloud viewer
python src/viz_critical_points.py      # Highlights PointNet critical points
python src/viz_pred_vs_label.py        # Side-by-side prediction vs. ground truth
```

---

### Running on Google Cloud VM

Besides our local resources, we also used a Google Cloud VM with the following configuration:

- **Machine type:** n1-standard-4 (4 vCPUs, 15 GB memory)
- **GPU:** 1 × NVIDIA T4
- **Boot disk:** 150 GB, Balanced persistent disk
- **Image:** Deep Learning on Linux — Deep Learning VM with CUDA 12.4 M129 (Debian 11, Python 3.10)

#### Environment setup

After connecting via SSH:
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Accept conda terms of service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Install git
sudo apt update
sudo apt install git
```

#### NVIDIA driver installation

The Deep Learning VM image ships without a driver by default. To install one:
```bash
sudo sed -i '/bullseye-backports/d' /etc/apt/sources.list

wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo sed -i 's/^deb https:\/\/deb.debian.org\/debian bullseye main$/deb https:\/\/deb.debian.org\/debian bullseye main contrib non-free/' /etc/apt/sources.list
sudo sed -i 's/^deb https:\/\/deb.debian.org\/debian bullseye-updates main$/deb https:\/\/deb.debian.org\/debian bullseye-updates main contrib non-free/' /etc/apt/sources.list

sudo apt update
sudo apt install -y nvidia-driver
sudo reboot
```

#### Verify GPU access

After creating the virtual environment similarly with miniconda:
```bash
conda create -n aidl-2026-project python=3.10 -y
conda activate aidl-2026-project
pip install --no-cache-dir -r requirements.txt
```

Verify GPU access:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## Experiment Tracking (W&B)

All training runs are tracked with [Weights & Biases](https://wandb.ai). The project is `dales-semantic-segmentation`, entity `aidl-3d-semantic-segmentation`.

### What is logged

**Per epoch during training** (`src/utils/trainer.py`):
- `Loss/Train`, `Loss/Validation`
- `Accuracy/Train`, `Accuracy/Validation`
- `mIoU/Train`, `mIoU/Validation`
- `Learning_Rate`

**During test** (`src/test.py`):
- `Loss/Test`, `Accuracy/Test`, `mIoU/Test`
- Per-class IoU: `IoU/Ground`, `IoU/Vegetation`, `IoU/Buildings`, `IoU/Vehicle`, `IoU/Utility`

**3D visualizations** are logged as W&B Tables with columns `epoch`, `GT` (ground truth point cloud), `Prediction`, `Errors`, `Uncertainty`. To keep storage manageable, only the first, middle, and last epoch are logged.

### Configuration

```yaml
wandb:
  enabled: true
  project: dales-semantic-segmentation
  entity: aidl-3d-semantic-segmentation
  mode: online   # online | offline | disabled
```

### Offline usage

If the VM has no outbound internet access, set `mode: offline`. Runs are cached locally under `wandb/` and can be synced afterward:

```bash
wandb sync wandb/
```

---

## PointNet Architecture

### Why PointNet for aerial LiDAR?

DALES is an aerial LiDAR dataset: each tile contains millions of unordered 3D points. PointNet operates directly on raw point sets (no grids, no images) so it preserves the full spatial precision of the original data. This matters when distinguishing small objects like vehicles and utility poles.

The key design insight is **permutation invariance via symmetric aggregation**: a shared MLP processes each point independently, and a max-pool over all points produces a global feature that is invariant to point ordering. This makes PointNet a natural fit for point clouds, which have no canonical order.

### Implementation (`src/models/pointnet.py`)

![PointNet architecture](figs/pointnet.jpg)

The model has three components:

**1. T-Net (TransformationNet)**

A mini-network that predicts a `k×k` transformation matrix to align the input (or its features) before processing. This is the spatial transformer idea applied to point clouds.

Architecture: shared MLP over points (`Conv1d`) with channels `k → 64 → 128 → 1024`, then global max-pool to a 1024-dim vector, then three FC layers `1024 → 512 → 256 → k×k`. The output is initialized as identity plus a learned offset, which stabilizes early training.

Two T-Nets are used: one on the raw 3D input (3×3) and one on the 64-dim feature space (64×64).

**2. PointNetBackbone**

| Stage | Operation | Output shape |
|---|---|---|
| Input transform | T-Net (3×3) applied to XYZ | `[B, N, 3]` |
| Shared MLP | Conv1d: `num_channels → 64 → 64` | `[B, 64, N]` |
| Feature transform | T-Net (64×64) applied to 64-dim features | `[B, 64, N]` |
| Point features | Conv1d: `64 → 128 → 1024` | `[B, 1024, N]` |
| Global feature | MaxPool over N points | `[B, 1024]` |
| Per-point concat | Local 64-dim + broadcast global 1024-dim | `[B, 1088, N]` |

**3. Segmentation head**

Takes the 1088-dim per-point feature vector and passes it through a shared MLP: `1088 → 512 → 256 → 128 → num_classes`, with dropout applied before the final layer. Output is passed through `log_softmax`.

**N.B.:** The original paper implements segmentation without dropout. We include a dropout parameter for experimentation but default it to 0 to match the paper's behavior.

**4. Classification head** (`PointNetClassification`)

Used for whole-cloud classification tasks (e.g., ModelNet and ShapeNet part classification). Takes only the **global feature** `[B, 1024]` from the backbone — not the per-point concatenation — and passes it through FC layers with batch norm: `1024 → 512 → 256 → num_classes`. Dropout is applied before the final layer, and the output is `log_softmax` over classes.

The key difference from the segmentation head is the input: classification uses the global max-pool feature alone (one prediction per cloud), while segmentation concatenates the global feature with per-point local features to produce one prediction per point.

---

## Data Augmentation

### Hypothesis

Aerial LiDAR scans are captured from above, so the same object (house, tree, car) can appear at any horizontal rotation depending on the flight path. Without augmentation, the model may learn orientation-specific patterns instead of actual shape.

Random scaling simulates the variation in point density across tiles caused by differences in scan overlap.

### Implementation (`src/models/dataset.py`)

Augmentation is applied only at training time (disabled for validation and test splits). It is implemented in `src/utils/dataset.py` and controlled through `config/default.yaml`:

```yaml
dataset:
  augmentation: true
  rotation_deg_max: 180.0    # Uniform sample from [-180°, +180°]
  scale_min: 0.9             # Uniform isotropic scale factor range
  scale_max: 1.1
```

Two transforms are applied per sample:

- **Random Z-axis rotation**: a rotation angle $\theta \sim \mathcal{U}(-180°, +180°)$ is sampled and applied to the XYZ coordinates only. Return number, and number-of-returns channels are unaffected.
- **Random isotropic scaling**: a scale factor $s \sim \mathcal{U}(0.9, 1.1)$ is applied to XYZ. This preserves the relative geometry of the point cloud while simulating density variation.

### Experiment Setup (TO BE ADDED)

### Results (TO BE ADDED)

### Conclusions (TO BE REVISED)

Z-rotation is the main augmentation for this dataset, it's cheap and directly matches the rotational symmetry of overhead scanning. It was kept on for all experiments.

---

## Focal Loss

### Hypothesis

DALES has severe class imbalance: Ground (53% of points), Vegetation (29%), and Buildings (17%) vastly outnumber Vehicle (<1%) and Utility (<1%). Standard NLL loss minimizes average per-point loss, so the gradient is dominated by easy, abundant Ground and Vegetation points. The model learns to classify those well while largely ignoring the rare classes. Focal Loss addresses this by dynamically down-weighting easy examples during training.

### Implementation (`src/utils/focal_loss.py`)

![Focal Loss](figs/focal_loss.png)

Based on [Lin et al., 2017 — Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002):

$$\text{FL}(p_t) = -\alpha_t \cdot (1 - p_t)^{\gamma} \cdot \log(p_t)$$

| Parameter | Role |
|---|---|
| $\alpha_t$ | Per-class weight (passed as a 5-element tensor). Controls baseline class importance independently of prediction confidence. |
| $\gamma$ | Focusing parameter (default: 2.0). When $\gamma = 0$, reduces to weighted cross-entropy. Higher $\gamma$ suppresses the contribution of easy, confidently classified examples and forces the loss gradient to come from hard misclassifications. |
| `ignore_index=-1` | Points with label `-1` (unknown/unlabeled) are masked out and excluded from loss computation entirely. |


Selected in config with:
```yaml
training:
  loss_function: focal_loss  # alternative: nll_loss
```
### Experiment Setup (TO BE ADDED)

### Results (TO BE ADDED)

### Conclusions (TO BE ADDED)

---

## Optimizer and Learning Rate

For training, we selected standard **Adam** optimizer (`torch.optim.Adam`) initialized with the `learning_rate` from config. By default this is `0.01`.

We also implemented a **cosine annealing** scheduler (`CosineAnnealingLR`) which decays the learning rate from the initial value down to a floor (`scheduler_min_lr`) following a cosine curve over the full training run:

![Learning Rate Scheduler](figs/learning_rate.png)

This avoids manually tuning step-decay milestones: the learning rate decreases smoothly to near zero by the final epoch, helping the model settle into a stable minimum.

### Configuration

```yaml
training:
  learning_rate: 0.01          # Initial LR passed to Adam
  scheduler_type: cosine       # Only supported option currently
  scheduler_min_lr: 0.00001    # Floor LR (eta_min in CosineAnnealingLR)
  num_epochs: 50               # Also used as T_max for the scheduler
```

---

## Class Weight Strategies

### Hypothesis

Even with Focal Loss, the $\alpha_t$ weights need to be set deliberately. An equal weight vector leaves the imbalance partially uncorrected, while an extreme inverse-frequency vector can destabilize training. The goal of these experiments is to find the weight strategy that achieves the best balance between overall IoU (dominated by Ground and Vegetation) and per-class IoU on the rare classes (Vehicle, Utility).

### Experiment Setup

Five weight strategies were evaluated, all producing a 5-element weight vector applied as $\alpha_t$ in Focal Loss (and as `weight` when using NLL Loss):

**Class frequencies:**

| Class | Index | Approx. frequency |
|---|---|---|
| Ground | 0 | 53% |
| Vegetation | 1 | 29% |
| Buildings | 2 | 17% |
| Vehicle | 3 | <1% |
| Utility | 4 | <1% |

**Strategies:**

1. **Square Root Inverse Frequency (`SQRT_INV_FREQ`)**: $w_c = 1 / \sqrt{f_c}$, then normalized. Mild correction that reduces dominance of common classes without completely suppressing them.

2. **Inverse Frequency (`INV_FREQ`)**: $w_c = 1 / f_c$, normalized. Stronger correction; rare classes receive much higher weights. Risks instability when class imbalance is extreme.

3. **Inverse Frequency Moderate (`INV_FREQ_MODERATE`)**: A capped version of inverse frequency where weights are clipped, preventing extremely small or large weights.

4. **Effective Number of Samples (`ENS_α`)**: Based on [Cui et al., 2019 — Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555). Weight $= (1 - \alpha) / (1 - \alpha^{f_c})$. Parameter $\alpha$ controls how aggressively rare samples are up-weighted. Three values were tested: $\alpha \in \{0.99999,\ 0.999999,\ 0.9999999\}$. Higher $\alpha$ gives a larger boost to rarer classes.

Example weight vector for ENS 0.99999 (used as default in `config/default.yaml`):

```yaml
loss_weights:
  - 0.0940   # Ground
  - 0.1197   # Vegetation
  - 0.1633   # Buildings
  - 2.0487   # Vehicle
  - 2.5743   # Utility
```

All other hyperparameters (learning rate, batch size, epochs, optimizer) were held constant across strategy runs to isolate the effect of the weights.

Below are the exact class weights computed for each strategy on our dataset. We did not test uniform weighting separately, as ENS 0.99999 already approximates it closely (weights range from 0.99 to 1.03).

| Strategy | Buildings | Ground | Utility | Vegetation | Vehicle |
|---|:---:|:---:|:---:|:---:|:---:|
| SQRT_INV_FREQ | 0.2553 | 0.3465 | 0.4482 | 1.8602 | 2.0897 |
| INV_FREQ | 0.0397 | 0.0731 | 0.1223 | 2.1066 | 2.6584 |
| INV_FREQ_MODERATE | 0.5000 | 0.5000 | 0.5000 | 2.1066 | 2.6584 |
| ENS 0.99999 | 0.9894 | 0.9894 | 0.9894 | 1.0049 | 1.0270 |
| ENS 0.999999 | 0.5272 | 0.5272 | 0.5276 | 1.5454 | 1.8727 |
| ENS 0.9999999 | 0.0940 | 0.1197 | 0.1633 | 2.0487 | 2.5743 |

### Results

The below results are provided for the holdout Test sample.

| Strategy | mIoU | IoU/Buildings | IoU/Ground | IoU/Utility | IoU/Vegetation | IoU/Vehicle |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| SQRT_INV_FREQ | 0.61 | 0.82 | 0.94 | 0.27 | 0.75 | 0.28 |
| INV_FREQ | 0.54 | 0.82 | 0.93 | $\color{red}{0.15}$ | 0.71 | $\color{red}{0.18}$ |
| INV_FREQ_MODERATE | 0.61 | 0.81 | 0.94 | 0.27 | 0.75 | 0.28 |
| ENS 0.99999 | $\color{green}{0.63}$ | $\color{green}{0.84}$ | $\color{green}{0.95}$ | 0.27 | $\color{green}{0.78}$ | $\color{green}{0.31}$ |
| ENS 0.999999 | 0.62 | 0.82 | 0.94 | $\color{green}{0.29}$ | 0.76 | 0.30 |
| ENS 0.9999999 | 0.58 | 0.82 | 0.94 | 0.19 | 0.73 | 0.23 |

### Conclusions

ENS 0.99999 achieves the best mIoU (0.63) and leads in four out of five classes. Its near-uniform weights suggest that aggressively upweighting rare classes is not necessary here and a gentle rebalancing is enough provided the focal loss implementation.

INV_FREQ performs worst overall (0.54), with the lowest scores on Utility and Vehicle. Its extreme weight ratio (~67× between Vehicle and Buildings) likely destabilizes training, hurting performance even on the rare classes it is meant to help.

SQRT_INV_FREQ and INV_FREQ_MODERATE produce identical mIoU (0.61) despite different weight distributions, indicating that moderate rebalancing strategies plateau at similar performance.

Utility remains the hardest class across all strategies (best: 0.29).

---

## Future Work

**1. GreenSegNet on DALES**

GreenSegNet is a lightweight semantic segmentation network designed for outdoor point clouds. Investigating whether it transfers to aerial LiDAR data from DALES would be a natural next step, particularly given its lower computational budget compared to PointNet.

**2. Pretrained 2D CNN features via BEV projection** -?

**3. Systematic hyperparameter search with Ray Tune**

`src/tune_ray.py` and `src/utils/trainer_for_ray.py` are already implemented. Running a structured sweep over learning rate, dropout, batch size, loss function, and ENS $\alpha$ values would yield further gains over the manual trial-and-error experiments documented here, without additional architectural changes.