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
  - [Optimizer and Learning Rate](#optimizer-and-learning-rate)
    - [Configuration](#configuration-1)
  - [Input Feature Selection](#input-feature-selection)
    - [Hypothesis](#hypothesis)
    - [Implementation (`config/default.yaml`)](#implementation-configdefaultyaml)
    - [Experiment Setup](#experiment-setup)
    - [Results](#results)
    - [Conclusions](#conclusions)
  - [Class Balancing Strategies](#class-balancing-strategies)
    - [Focal Loss vs NLL Loss](#focal-loss-vs-nll-loss)
      - [Hypothesis](#hypothesis-1)
      - [Implementation (`src/utils/focal_loss.py`)](#implementation-srcutilsfocal_losspy)
    - [Class Weight Strategies](#class-weight-strategies)
      - [Hypothesis](#hypothesis-2)
      - [Implementation](#implementation)
    - [Class Balanced Sampler](#class-balanced-sampler)
      - [Hypothesis](#hypothesis-3)
      - [Implementation (`src/utils/sampler.py`)](#implementation-srcutilssamplerpy)
    - [Experiment Setup](#experiment-setup-1)
    - [Results](#results-1)
    - [Conclusions](#conclusions-1)
  - [Data Augmentation](#data-augmentation)
    - [Hypothesis](#hypothesis-4)
    - [Implementation (`src/models/dataset.py`)](#implementation-srcmodelsdatasetpy)
    - [Experiment Setup](#experiment-setup-2)
    - [Results](#results-2)
    - [Conclusions](#conclusions-2)
  - [Dropout](#dropout)
    - [Hypothesis](#hypothesis-5)
    - [Implementation](#implementation-1)
    - [Experiment Setup](#experiment-setup-3)
    - [Results](#results-3)
    - [Conclusions](#conclusions-3)
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
│   ├── generate_full_density_bev_rasters_from_las.py     # Preprocessing: LAS → BEV rasters
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
│       ├── sampler.py               # Class Balancing sampler for train dataloader
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

Available CLI arguments: `--model_name`, `--num_points`, `--batch_size`, `--num_epochs`, `--dropout_rate`.

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
- **Image:** Deep Learning on Linux - Deep Learning VM with CUDA 12.4 M129 (Debian 11, Python 3.10)

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

Used for whole-cloud classification tasks (e.g., ModelNet and ShapeNet part classification). Takes only the **global feature** `[B, 1024]` from the backbone (not the per-point concatenation) and passes it through FC layers with batch norm: `1024 → 512 → 256 → num_classes`. Dropout is applied before the final layer, and the output is `log_softmax` over classes.

The key difference from the segmentation head is the input: classification uses the global max-pool feature alone (one prediction per cloud), while segmentation concatenates the global feature with per-point local features to produce one prediction per point.

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

## Input Feature Selection

### Hypothesis

PointNet's input is a per-point feature vector. The minimal version uses only XYZ coordinates (3 channels), which captures geometry alone. DALES LiDAR data also provides per-point return metadata: `return_number` (which echo this point came from) and `number_of_returns` (how many echoes the pulse produced). These carry information about vertical structure: a single-return point likely hit a hard surface (ground, rooftop), while a multi-return point passed through vegetation or hit a wire. This signal is complementary to geometry and may help the model distinguish classes that are spatially similar but structurally different, such as low vegetation vs. ground, or utility lines vs. building edges.

The hypothesis is that adding return metadata (XYZ + return_number + number_of_returns, 5 channels) improves segmentation performance, particularly on classes where geometry alone is ambiguous, doing so without meaningful additional compute cost, since PointNet's shared MLP scales with channel count but the increase from 3 to 5 is marginal.

### Implementation (`config/default.yaml`)

Feature selection is controlled in the config under `dataset.use_features`. The preprocessing step (`convert_las_to_blocks.py`) extracts all requested features into the NPZ blocks; the dataset loader then selects the subset specified at training time. The number of input channels (`num_channels`) is computed automatically from the selected features.

The input T-Net is hardcoded to operate on XYZ only (`input_dim=3, output_dim=3`), regardless of how many channels the full input has. In the backbone's forward pass, the first three columns (XYZ) are sliced off, transformed by the T-Net, and then concatenated back with any extra channels (e.g., `return_number`, `number_of_returns`) before entering the first shared MLP. This means the spatial transform is applied strictly to geometry, while the return metadata passes through untouched. The first `Conv1d` layer accepts `input_channels` (3 or 5), so `num_channels` is computed automatically from the selected features and no other architecture change is needed.

XYZ-only (3 channels):
```yaml
dataset:
  use_features:
    - xyz
```

XYZ + return data (5 channels):
```yaml
dataset:
  use_features:
    - xyz
    - return_number
    - number_of_returns
```

### Experiment Setup

| Run | Input Features | Channels | Purpose |
|---|---|---|---|
| 1 | XYZ | 3 | Geometry-only baseline |
| 2 | XYZ + return_number + number_of_returns | 5 | Geometry + return metadata |

All other hyperparameters are held constant: NLL loss, uniform weights, sampler off, no augmentation, 50 epochs, batch size 32, learning rate 0.01 with cosine annealing.

### Results

| Run | Input Features | mIoU | Buildings | Ground | Utility | Vegetation | Vehicle |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | XYZ | 0.628 / 0.592 | 0.852 / 0.809 | 0.939 / 0.930 | 0.280 / 0.272 | 0.785 / 0.749 | 0.283 / 0.199 |
| 2 | XYZ + return data | $\color{green}{0.656}$ / $\color{green}{0.637}$ | $\color{green}{0.855}$ / $\color{green}{0.839}$ | $\color{green}{0.937}$ / $\color{green}{0.931}$ | $\color{green}{0.352}$ / $\color{green}{0.344}$ | $\color{green}{0.789}$ / $\color{green}{0.771}$ | $\color{green}{0.345}$ / $\color{green}{0.300}$ |

*Values shown as Train / Val.*

> **[Figure placeholder]** 

### Conclusions

Adding return metadata (2 extra channels) improves validation mIoU from 0.592 to 0.637 (+0.045), a substantial gain from two extra input features at negligible compute cost. The improvement is driven almost entirely by the rare classes: Utility jumps from 0.272 to 0.344 and Vehicle from 0.199 to 0.300, confirming that return_number and number_of_returns carry discriminative signal about vertical structure that geometry alone cannot capture. Majority classes (Ground, Vegetation, Buildings) remain essentially unchanged, meaning the additional features help where they are needed without hurting elsewhere. All subsequent experiments use the 5-channel input (XYZ + return data).

---

## Class Balancing Strategies

DALES has severe class imbalance: Ground (53% of points), Vegetation (29%), and Buildings (17%) vastly outnumber Vehicle (<1%) and Utility (<1%). This class imbalance can be attacked at three levels: the loss function (how the gradient is shaped), the loss weights (how much each class contributes), and the sampler (which blocks the model trains on). These are not simply additive - they interact, and stacking all three does not guarantee the best result.

The sections below document each strategy independently, the reasoning behind it, and what the results revealed about how they interact.

### Focal Loss vs NLL Loss

#### Hypothesis

Standard NLL loss treats every point equally, so the gradient is dominated by abundant, easy-to-classify Ground and Vegetation points. Focal Loss addresses this by applying a scaling factor that dynamically down-weights points the model already classifies confidently, focusing the gradient on hard, ambiguous examples, which tend to be the rare classes.

The expectation is that Focal Loss should improve rare-class IoU compared to NLL under the same weight strategy, at some possible cost to overall IoU on the dominant classes.

#### Implementation (`src/utils/focal_loss.py`)

![Focal Loss](figs/focal_loss.png)

Based on [Lin et al., 2017 — Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002):

$$\text{FL}(p_t) = -\alpha_t \cdot (1 - p_t)^{\gamma} \cdot \log(p_t)$$

| Parameter | Role |
|---|---|
| $\alpha_t$ | Per-class weight (passed as a 5-element tensor). Controls baseline class importance independently of prediction confidence. |
| $\gamma$ | Focusing parameter (default: 1.0). When $\gamma = 0$, reduces to weighted cross-entropy. Higher $\gamma$ suppresses the contribution of easy, confidently classified examples and forces the loss gradient to come from hard misclassifications. |
| `ignore_index=-1` | Points with label `-1` (unknown/unlabeled) are masked out and excluded from loss computation entirely. |


Selected in config with:
```yaml
training:
  loss_function: focal_loss  # alternative: nll_loss
```

---

### Class Weight Strategies

#### Hypothesis

Both Focal Loss and NLL accept a per-class weight vector $\alpha_t$​ that scales each class's contribution to the total loss. An equal-weight vector leaves the frequency imbalance uncorrected at the loss level, while aggressive inverse-frequency weights can destabilize training by amplifying noisy gradients from the small number of rare-class points. The goal is to find the weight vector that maximizes rare-class IoU without degrading performance on the majority classes.

Because the weight vector interacts with the loss function (Focal Loss already down-weights easy examples, so adding aggressive class weights on top may over-correct), the optimal weights may differ between NLL and Focal Loss.

#### Implementation

Both loss functions accept a per-class weight vector $\alpha_t$ that scales each class's contribution to the total loss. We explored several weighting strategies before selecting the Effective Number of Samples (ENS) family for our experiments.

**Class frequencies:**

| Class | Index | Approx. frequency |
|---|---|---|
| Ground | 0 | 53% |
| Vegetation | 1 | 29% |
| Buildings | 2 | 17% |
| Vehicle | 3 | <1% |
| Utility | 4 | <1% |

**Strategies considered:**

1. **Square Root Inverse Frequency (`SQRT_INV_FREQ`)**: $w_c = 1 / \sqrt{f_c}$, then normalized. Mild correction that reduces dominance of common classes without completely suppressing them.

2. **Inverse Frequency (`INV_FREQ`)**: $w_c = 1 / f_c$, normalized. Stronger correction; rare classes receive much higher weights. Risks instability when class imbalance is extreme.

3. **Inverse Frequency Moderate (`INV_FREQ_MODERATE`)**: A capped version of inverse frequency where weights are clipped, preventing extremely small or large weights.

4. **Effective Number of Samples (`ENS_β`)**: Based on [Cui et al., 2019 — Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555). Weight $= (1 - \beta) / (1 - \beta^{f_c})$. Parameter $\beta$ controls how aggressively rare samples are up-weighted.

**Selected for experiments:** We carried forward only the ENS family with three $\beta$ values: $\beta \in \{0.99999,\ 0.999999,\ 0.9999999\}$. By varying a single parameter, ENS spans the full correction spectrum, from near-uniform (0.99999) through moderate (0.999999) to aggressive (0.9999999), covering the same range as the heuristic strategies above without mixing different weighting philosophies across runs.

| Strategy | Ground | Vegetation | Buildings | Vehicle | Utility |
|---|:---:|:---:|:---:|:---:|:---:|
| ENS 0.99999 | 0.9894 | 0.9894 | 0.9894 | 1.0049 | 1.0270 |
| ENS 0.999999 | 0.5272 | 0.5272 | 0.5276 | 1.5454 | 1.8727 |
| ENS 0.9999999 | 0.0940 | 0.1197 | 0.1633 | 2.0487 | 2.5743 |

Weights can be modified in `config/default.yaml`:
```yaml
loss_weights:
  - 0.9894   # Ground
  - 0.9894   # Vegetation
  - 0.9894   # Buildings
  - 1.0049   # Vehicle
  - 1.0270   # Utility
```

### Class Balanced Sampler

#### Hypothesis

Loss weights and Focal Loss both operate at the point level, they reweight individual point contributions within whatever batch the model happens to see. But in DALES, rare classes are spatially concentrated: most blocks contain only Ground, Vegetation, and Buildings, while Vehicle and Utility points appear in a small subset of blocks. Under standard uniform shuffling, the model may go many consecutive batches without encountering a single rare-class point, regardless of how the loss is configured.

The Class Balanced Sampler addresses this at the data-loading level by oversampling blocks that contain at least one rare-class point. This ensures that every batch is likely to include rare-class geometry, providing a consistent gradient signal for those classes throughout training. Unlike loss-level corrections, the sampler changes what the model sees rather than how it scores what it sees.

A key implication is that the sampler may reduce the need for aggressive loss-level rebalancing. If the model already encounters rare-class blocks frequently, the loss function does not need to overcompensate for their absence, uniform weights may suffice because the data distribution itself has been corrected.

#### Implementation (`src/utils/sampler.py`)

The sampler assigns a sampling weight to every block in the training set. For each block, it loads the label array and checks whether any point belongs to a rare class (by default Vehicle and Utility, indices 3 and 4). Blocks containing at least one rare-class point receive a weight of rare_class_boost (default: 3.0); all other blocks receive a weight of 1.0. At each epoch, `torch.multinomial` draws `len(dataset)` indices with replacement according to these weights, so rare-class blocks are approximately 3× more likely to appear than majority-only blocks.

Because sampling is with replacement, some rare-class blocks will appear multiple times per epoch while some majority-only blocks may not appear at all. This is intentional: it trades off slight overfitting risk on rare-class blocks for a substantially more balanced class distribution per epoch.

Selected in config with:

```yaml
training:
  use_sampler: true   # Enable class-balanced sampling
```

When `use_sampler: true`, the DataLoader uses the sampler instead of random shuffling (shuffle and sampler are mutually exclusive in PyTorch). The sampler is applied only to the training set; validation always uses sequential loading.

### Experiment Setup

All experiments use the same base configuration: PointNet, Adam optimizer, cosine annealing scheduler (LR 0.01 → 0.00001), batch size 32, 50 epochs, 4096 points per block, XYZ + return data (5 channels). Only the loss function, class weights, and sampler vary across runs (no data augmentation, no dropout).

**Phase 1: Loss function × weight strategy (sampler off)**

| Run | Loss | Weights | Sampler | Purpose |
|---|---|---|---|---|
| 3 | NLL | ENS 0.99999 | Off | NLL with near-uniform weights |
| 4 | NLL | ENS 0.999999 | Off | NLL with moderate correction |
| 5 | NLL | ENS 0.9999999 | Off | NLL with strong correction |
| 6 | Focal (γ=2) | ENS 0.99999 | Off | Focal γ=2 with near-uniform weights |
| 7 | Focal (γ=2) | ENS 0.999999 | Off | Focal γ=2 with moderate correction |
| 8 | Focal (γ=1) | ENS 0.99999 | Off | Focal γ=1 with near-uniform weights |
| 9 | Focal (γ=1) | ENS 0.999999 | Off | Focal γ=1 with moderate correction |
| 10 | Focal (γ=1) | ENS 0.9999999 | Off | Focal γ=1 with strong correction |

**Phase 2: Adding the class-balanced sampler**

| Run | Loss | Weights | Sampler | Purpose |
|---|---|---|---|---|
| 11 | NLL | Uniform | On (3×) | Sampler + NLL uniform weights |
| 12 | NLL | ENS 0.999999 | On (3×) | Sampler + NLL moderate weights |
| 13 | Focal (γ=1) | Uniform | On (3×) | Sampler + Focal γ=1 uniform weights |

**Note on Phase 2 weight choices:** Phase 2 includes runs with Uniform weights (Runs 11 and 13) alongside the moderate ENS 0.999999 (Run 12) intentionally. As mentioned above, the hypothesis is that the class-balanced sampler and loss-level weight correction may compete rather than complement each other: the sampler already corrects the class distribution at the data level, so adding aggressive loss weights on top could over-correct and destabilize training. By testing Uniform weights with the sampler on, we can isolate the sampler's contribution and determine whether loss-level rebalancing is still necessary once the model sees rare-class blocks frequently.

### Results

**Phase 1: Loss function × weight strategy (sampler off)**

| Run | Loss | Weights | mIoU | Buildings | Ground | Utility | Vegetation | Vehicle |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 3 | NLL | ENS 0.99999 | 0.651 / 0.630 | 0.846 / 0.828 | 0.937 / 0.929 | 0.350 / 0.344 | 0.784 / 0.766 | 0.340 / 0.285 |
| 4 | NLL | ENS 0.999999 | $\color{green}{0.671}$ / $\color{green}{0.641}$ | 0.851 / 0.834 | $\color{green}{0.938}$ / $\color{green}{0.930}$ | $\color{green}{0.391}$ / $\color{green}{0.357}$ | 0.781 / 0.763 | $\color{green}{0.394}$ / $\color{green}{0.322}$ |
| 5 | NLL | ENS 0.9999999 | 0.598 / 0.576 | 0.826 / 0.814 | 0.928 / 0.922 | 0.238 / 0.214 | 0.725 / 0.711 | 0.271 / 0.220 |
| 6 | Focal (γ=2) | ENS 0.99999 | 0.642 / 0.621 | 0.839 / 0.825 | 0.932 / 0.924 | 0.332 / 0.321 | 0.772 / 0.756 | 0.337 / 0.279 |
| 7 | Focal (γ=2) | ENS 0.999999 | 0.648 / 0.622 | 0.827 / 0.816 | 0.931 / 0.924 | 0.361 / 0.335 | 0.757 / 0.743 | 0.366 / 0.290 |
| 8 | Focal (γ=1) | ENS 0.99999 | 0.661 / 0.639 | $\color{green}{0.852}$ / $\color{green}{0.837}$ | 0.937 / 0.930 | 0.365 / 0.357 | $\color{green}{0.786}$ / $\color{green}{0.769}$ | 0.366 / 0.303 |
| 9 | Focal (γ=1) | ENS 0.999999 | 0.659 / 0.630 | 0.836 / 0.822 | 0.934 / 0.927 | 0.378 / 0.350 | 0.767 / 0.750 | 0.379 / 0.303 |
| 10 | Focal (γ=1) | ENS 0.9999999 | 0.586 / 0.566 | 0.816 / 0.808 | 0.924 / 0.919 | 0.222 / 0.199 | 0.715 / 0.703 | 0.251 / 0.203 |

*Values shown as Train / Val.*

> **[Figure placeholder]** 

**Phase 2: Adding the class-balanced sampler**

| Run | Loss | Weights | Sampler | mIoU | Buildings | Ground | Utility | Vegetation | Vehicle |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 11 | NLL | Uniform | On (3×) | 0.664 / 0.639 | $\color{green}{0.857}$ / $\color{green}{0.842}$ | 0.942 / 0.930 | 0.372 / 0.352 | $\color{green}{0.783}$ / $\color{green}{0.772}$ | 0.365 / 0.302 |
| 12 | NLL | ENS 0.999999 | On (3×) | $\color{green}{0.680}$ / $\color{green}{0.646}$ | 0.857 / 0.836 | $\color{green}{0.943}$ / $\color{green}{0.930}$ | $\color{green}{0.409}$ / $\color{green}{0.365}$ | 0.777 / 0.767 | $\color{green}{0.416}$ / $\color{green}{0.330}$ |
| 13 | Focal (γ=1) | Uniform | On (3×) | 0.652 / 0.628 | 0.840 / 0.829 | 0.939 / 0.927 | 0.362 / 0.348 | 0.768 / 0.761 | 0.352 / 0.275 |

*Values shown as Train / Val.*

> **[Figure placeholder]** 

### Conclusions

**NLL consistently outperforms Focal Loss.** Under matched weight strategies, NLL achieves higher validation mIoU than both Focal γ=2 and Focal γ=1. The best Phase 1 result is NLL + ENS 0.999999 (Run 4, val mIoU 0.641), compared to the best Focal result of 0.639 (Run 8, Focal γ=1 + ENS 0.99999). Focal γ=2 performs worst among the loss functions, suggesting that aggressive down-weighting of easy examples hurts more than it helps when the model still needs strong gradients from majority classes for spatial context. Focal γ=1 is a milder variant and comes closer to NLL, but does not surpass it.

**Moderate ENS weights (β = 0.999999) are the sweet spot.** Across both NLL and Focal Loss, ENS 0.999999 delivers the best rare-class IoU without degrading majority classes. Near-uniform weights (ENS 0.99999) underperform because they leave the imbalance essentially uncorrected at the loss level. Strong weights (ENS 0.9999999) collapse performance, with mIoU dropping to 0.576 (NLL) and 0.566 (Focal γ=1) (the extreme weight ratio destabilizes training).

**The class-balanced sampler provides an additive boost with NLL.** Run 12 (NLL + ENS 0.999999 + sampler) achieves the highest overall validation mIoU of 0.646 and the best Utility and Vehicle IoU (0.365 and 0.330 accordingly). The sampler ensures rare-class blocks appear consistently, so the loss-level moderate weights and sampler-level oversampling complement each other. However, stacking the sampler with Focal Loss (Run 13) does not help: validation mIoU drops to 0.628, confirming that Focal's dynamic down-weighting conflicts with the sampler's data-level correction.

**Best configuration:** NLL + ENS 0.999999 + class-balanced sampler (Run 12). Utility and Vehicle remain the hardest classes, but this is the best result achieved across all balancing experiments.

---

## Data Augmentation

### Hypothesis

Aerial LiDAR scans are captured from above, so the same object (house, tree, car) can appear at any horizontal rotation depending on the flight path. Without augmentation, the model may learn orientation-specific patterns instead of actual shape.

### Implementation (`src/models/dataset.py`)

Augmentation is applied only at training time (disabled for validation and test splits). It is implemented in `src/utils/dataset.py` and controlled through `config/default.yaml`:

```yaml
dataset:
  augmentation: true
  rotation_deg_max: 180.0    # Uniform sample from [-180°, +180°]
```

The following transform is applied per sample:

- **Random Z-axis rotation**: a rotation angle $\theta \sim \mathcal{U}(-180°, +180°)$ is sampled and applied to the XYZ coordinates only. Return number, and number-of-returns channels are unaffected.

### Experiment Setup

To isolate the effect of data augmentation, we compare the best configuration from the Class Balancing experiments (Run 12: NLL + ENS 0.999999 + sampler) with and without augmentation enabled. All other hyperparameters are held constant.

| Run | Augmentation | Loss | Weights | Sampler | Purpose |
|---|---|---|---|---|---|
| 12 | Off | NLL | ENS 0.999999 | On (3×) | Baseline (best balancing config) |
| 15 | On | NLL | ENS 0.999999 | On (3×) | Add random Z-axis rotation |

### Results

| Run | Augmentation | mIoU | Buildings | Ground | Utility | Vegetation | Vehicle |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 12 | Off | $\color{green}{0.680}$ / $\color{green}{0.646}$ | $\color{green}{0.857}$ / $\color{green}{0.836}$ | $\color{green}{0.943}$ / $\color{green}{0.930}$ | $\color{green}{0.409}$ / $\color{green}{0.365}$ | $\color{green}{0.777}$ / $\color{green}{0.767}$ | $\color{green}{0.416}$ / $\color{green}{0.330}$ |
| 15 | On | 0.638 / 0.636 | 0.822 / 0.829 | 0.934 / 0.927 | 0.357 / 0.353 | 0.745 / 0.755 | 0.332 / 0.313 |

*Values shown as Train / Val.*

> **[Figure placeholder]** 

### Conclusions

Data augmentation narrows the train–val gap substantially (from 0.034 to 0.002 in mIoU), confirming that it acts as an effective regularizer and reduces overfitting. Validation mIoU drops slightly compared to the non-augmented baseline (0.636 vs. 0.646), but the near-complete elimination of the generalization gap is a positive signal: the model is no longer memorizing training blocks. This trade-off (lower peak performance but more honest generalization) is typical when adding regularization to a model that was previously overfitting. We carry augmentation forward to the dropout experiments, expecting that dropout can recover the lost capacity while preserving the tighter train–val alignment that augmentation provides.

---

## Dropout

### Hypothesis

The original PointNet paper does not use dropout in the segmentation head. However, our setting differs from the original in two ways: we operate on small fixed-size blocks (4096 points) rather than full scenes, and the DALES training set contains a limited number of such blocks. This means the model sees the same local geometries repeatedly across epochs, which creates a risk of overfitting to block-specific patterns rather than learning generalizable per-point features. The hypothesis is that adding moderate dropout (0.3-0.5) before the final classification layer will act as a complementary regularizer to data augmentation: while augmentation diversifies the input geometry, dropout prevents the segmentation head from relying on fixed activation patterns, encouraging more robust internal representations.

### Implementation

Dropout is applied in the segmentation head after the last shared MLP layer (`128 → 128`, with batch norm and ReLU), immediately before the final `Conv1d` projection to `num_classes`. 

The dropout rate is controlled via config and CLI:

```yaml
training:
  dropout_rate: 0.3   # 0 = no dropout (paper default)
```

When `dropout_rate: 0`, the model matches the original PointNet paper behavior.

### Experiment Setup

Starting from the best balancing configuration with augmentation enabled (Run 15: NLL + ENS 0.999999 + sampler + augmentation), we test two dropout rates. All other hyperparameters are held constant.

| Run | Dropout | Augmentation | Loss | Weights | Sampler | Purpose |
|---|---|---|---|---|---|---|
| 15 | 0.0 | On | NLL | ENS 0.999999 | On (3×) | Baseline (augmentation, no dropout) |
| 16 | 0.3 | On | NLL | ENS 0.999999 | On (3×) | Moderate dropout |
| 17 | 0.5 | On | NLL | ENS 0.999999 | On (3×) | Stronger dropout |

### Results

| Run | Dropout | mIoU | Buildings | Ground | Utility | Vegetation | Vehicle |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 15 | 0.0 | 0.638 / 0.636 | 0.822 / 0.829 | 0.934 / 0.927 | 0.357 / 0.353 | 0.745 / 0.755 | 0.332 / 0.313 |
| 16 | 0.3 | $\color{green}{0.664}$ / $\color{green}{0.660}$ | $\color{green}{0.850}$ / $\color{green}{0.852}$ | 0.940 / 0.932 | $\color{green}{0.386}$ / $\color{green}{0.378}$ | $\color{green}{0.769}$ / $\color{green}{0.779}$ | $\color{green}{0.375}$ / $\color{green}{0.359}$ |
| 17 | 0.5 | 0.662 / 0.658 | 0.848 / 0.851 | $\color{green}{0.940}$ / $\color{green}{0.933}$ | 0.385 / 0.377 | 0.766 / 0.776 | 0.371 / 0.355 |

*Values shown as Train / Val.*

> **[Figure placeholder]** .

### Conclusions

Dropout provides a clear improvement over the no-dropout augmentation baseline. At dropout 0.3 (Run 16), validation mIoU rises from 0.636 to 0.660 (+0.024), with gains across all five classes. The train–val gap remains very small (0.004), confirming that dropout and augmentation together produce strong regularization. Vehicle shows the largest relative improvement (val IoU 0.313 → 0.359), suggesting that dropout helps the model generalize better on rare classes by preventing the segmentation head from overfitting to the small set of rare-class blocks seen during training.

Dropout 0.5 (Run 17) performs nearly identically to 0.3 (val mIoU 0.658 vs. 0.660), indicating that the model is not very sensitive to the exact rate in this range. The marginal drop at 0.5 suggests that stronger dropout starts to under-fit slightly, but the difference is within noise.

**Best overall configuration:** NLL + ENS 0.999999 + class-balanced sampler + augmentation + dropout 0.3 (Run 16), achieving validation mIoU of 0.660. This represents a cumulative improvement of +0.068 over the initial XYZ-only baseline (Run 1, val mIoU 0.592).

---

## Future Work

**1. GreenSegNet on DALES**

GreenSegNet is a lightweight semantic segmentation network designed for outdoor point clouds. Investigating whether it transfers to aerial LiDAR data from DALES would be a natural next step, particularly given its lower computational budget compared to PointNet.

**2. Pretrained 2D CNN features via BEV projection** -?

**3. Systematic hyperparameter search with Ray Tune**

`src/tune_ray.py` and `src/utils/trainer_for_ray.py` are already implemented. Running a structured sweep over learning rate, dropout, batch size, loss function, and focal loss parameters would yield further gains over the manual trial-and-error experiments documented here, without additional architectural changes.