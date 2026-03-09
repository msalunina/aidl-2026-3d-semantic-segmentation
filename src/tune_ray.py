# src/tune_ray.py
# Use example: python -m src.tune_ray --num-samples 5 --max-epochs 10 --cpus 4 --gpus 1 --ray-tmp C:\rtmp
#
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import os
import sys
import random
import numpy as np
import torch

from ray import tune
from ray.tune.schedulers import ASHAScheduler

from src.utils.config_parser import ConfigParser
from src.utils.dataset import DALESDataset
from torch.utils.data import DataLoader
from src.utils.trainer_for_ray import train_model_segmentation
from src.utils.focal_loss import FocalLoss
from src.utils.sampler import ClassBalancedSampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Weight-computation helpers (mirrors compute_class_frequencies.py but inline
# to avoid module-level side effects when that file is imported by Ray workers)
# ---------------------------------------------------------------------------

def compute_focal_weights(
    counter,
    method: str = "sqrt_inv_freq",
    beta: float = 0.9999,
    num_classes: int = 5,
) -> np.ndarray:
    """Compute per-class weights suitable for use with FocalLoss.

    Methods
    -------
    - 'sqrt_inv_freq': Square root of inverse frequency (recommended for Focal Loss)
    - 'moderate': Moderate inverse frequency with clipping
    - 'effective_num': Effective number of samples (Cui et al. 2019)
    - 'uniform': All ones
    """
    if method == "uniform":
        return np.ones(num_classes, dtype=np.float64)

    freqs = np.array([counter.get(i, 0) for i in range(num_classes)], dtype=np.float64)
    freqs = freqs / freqs.sum()

    if method == "sqrt_inv_freq":
        weights = np.sqrt(1.0 / (freqs + 1e-6))
        weights = weights / weights.mean()

    elif method == "moderate":
        weights = 1.0 / (freqs + 1e-6)
        weights = weights / weights.mean()
        weights = np.clip(weights, 0.5, 5.0)

    elif method == "effective_num":
        counts = np.array([counter.get(i, 0) for i in range(num_classes)], dtype=np.float64)
        effective_num = (1 - np.power(beta, counts)) / (1 - beta)
        weights = 1.0 / (effective_num + 1e-6)
        weights = weights / weights.sum() * num_classes

    else:
        raise ValueError(f"Unknown weight method: {method}")

    return weights


def precompute_class_weights(
    data_dir: str | Path,
    ignore_label: int,
    num_classes: int = 5,
) -> list[list[float]]:
    """Scan all .npz files in *data_dir* once and return a list of weight
    vectors (one per scheme) ready for use as tune.choice options.

    Returned order
    --------------
    0  sqrt_inv_freq
    1  moderate
    2  effective_num  beta=1 - 1e-5
    3  effective_num  beta=1 - 1e-6
    4  effective_num  beta=1 - 1e-7
    5  uniform
    """
    from collections import Counter

    files = list(Path(data_dir).glob("*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files found in {data_dir}")

    counter: Counter = Counter()
    total_points = 0
    for f in files:
        data = np.load(f)
        labels = data["labels"]
        labels = labels[labels != ignore_label]
        total_points += labels.size
        counter.update(labels.tolist())

    print(f"[precompute_class_weights] scanned {len(files)} blocks, {total_points:,} labeled points")

    schemes = [
        ("sqrt_inv_freq", None),
        ("moderate",      None),
        ("effective_num", 1 - 1e-5),
        ("effective_num", 1 - 1e-6),
        ("effective_num", 1 - 1e-7),
        ("uniform",       None),
    ]

    weight_options: list[list[float]] = []
    print("\n[precompute_class_weights] computed weight options:")
    for method, beta in schemes:
        kw = {} if beta is None else {"beta": beta}
        w = compute_focal_weights(counter, method=method, num_classes=num_classes, **kw)
        label = method if beta is None else f"{method}(beta={beta})"
        rounded = [round(float(x), 6) for x in w]
        print(f"  {label:35s}: {rounded}")
        weight_options.append(rounded)

    return weight_options


def load_base_config():
    """
    Loads existing YAML config via ConfigParser, and returns (cfg, repo_root_path).

    IMPORTANT for Ray:
    - Ray workers have extra argv flags (e.g., --node-ip-address=...).
    - If ConfigParser internally uses argparse.parse_args(), it will crash.
    - So we temporarily sanitizing sys.argv while loading the config.
    """
    repo_root = Path(__file__).resolve().parents[1]  # src/ -> repo root
    config_path = repo_root / "config" / "default.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config_parser = ConfigParser(
        default_config_path=str(config_path),  # ABSOLUTE PATH
        parser=argparse.ArgumentParser(description="Ray Tune HPO for DALES Segmentation"),
    )

    old_argv = sys.argv[:]
    try:
        # Strip Ray-injected args so ConfigParser/argparse doesn't explode
        sys.argv = [old_argv[0]]
        cfg = config_parser.load()
    finally:
        sys.argv = old_argv

    return cfg, repo_root


def trainable(trial_cfg: dict):
    cfg, base_path = load_base_config()

    # Ensure the repo root and src/ are on sys.path so that intra-package
    # imports like `from models.img_encoder import ...` work in Ray workers,
    # which do not inherit the parent process's sys.path.
    for _p in [str(base_path), str(base_path / "src")]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    # Resolve relative paths to absolute
    def _abs(p: str) -> str:
        path = Path(p)
        return str(path if path.is_absolute() else base_path / path)

    cfg.model_data_path = _abs(cfg.model_data_path)
    cfg.image_data_path = _abs(cfg.image_data_path)

    # Disable W&B + snapshots inside HPO trials
    cfg.wandb_enabled = False
    cfg.save_snapshots = False

    # Override hyperparameters
    cfg.learning_rate = float(trial_cfg["learning_rate"])
    cfg.batch_size = int(trial_cfg["batch_size"])
    cfg.num_epochs = int(trial_cfg["num_epochs"])

    focal_gamma = float(trial_cfg["focal_gamma"])
    rare_class_boost = float(trial_cfg["rare_class_boost"])

    # Override loss weights if provided by the tuner
    if "loss_weights" in trial_cfg:
        cfg.loss_weights = list(trial_cfg["loss_weights"])

    # Optional: tune cosine scheduler eta_min if provided
    if getattr(cfg, "scheduler_type", None) == "cosine" and "scheduler_min_lr" in trial_cfg:
        cfg.scheduler_min_lr = float(trial_cfg["scheduler_min_lr"])

    # Optionally tune augmentation severity (only matters if augmentation is enabled)
    if getattr(cfg, "dataset_augmentation", False):
        cfg.dataset_rotation_deg_max = float(trial_cfg["rotation_deg_max"])
        cfg.dataset_scale_min = float(trial_cfg["scale_min"])
        cfg.dataset_scale_max = float(trial_cfg["scale_max"])

    # Determinism
    cfg.dataset_seed = int(trial_cfg.get("seed", cfg.dataset_seed))
    set_seed(cfg.dataset_seed)
    device = set_device()

    # Datasets / loaders
    train_dataset = DALESDataset(
        data_dir=f"{cfg.model_data_path}/train",
        images_dir=f"{cfg.image_data_path}/train",
        split="train",
        use_features=cfg.dataset_use_features,
        num_points=cfg.train_num_points,
        normalize=cfg.dataset_normalize,
        augmentation=cfg.dataset_augmentation,
        rotation_deg_max=cfg.dataset_rotation_deg_max,
        scale_min=cfg.dataset_scale_min,
        scale_max=cfg.dataset_scale_max,
        train_ratio=cfg.dataset_train_ratio,
        val_ratio=cfg.dataset_val_ratio,
        seed=cfg.dataset_seed,
    )
    val_dataset = DALESDataset(
        data_dir=f"{cfg.model_data_path}/train",
        images_dir=f"{cfg.image_data_path}/train",
        split="val",
        use_features=cfg.dataset_use_features,
        num_points=cfg.train_num_points,
        normalize=cfg.dataset_normalize,
        train_ratio=cfg.dataset_train_ratio,
        val_ratio=cfg.dataset_val_ratio,
        seed=cfg.dataset_seed,
    )

    train_sampler = ClassBalancedSampler(
        train_dataset,
        rare_classes=[3, 4],
        rare_class_boost=rare_class_boost,
        verbose=False,
    )

    num_workers = int(getattr(cfg, "num_workers", 0))
    pin_memory = bool(getattr(cfg, "pin_memory", torch.cuda.is_available()))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Model
    if cfg.model_name == "pointnet":
        from src.models.pointnet import PointNetSegmentation

        model = PointNetSegmentation(
            num_classes=cfg.num_classes,
            input_channels=cfg.num_channels,
            dropout=cfg.dropout_rate,
        ).to(device)
    elif cfg.model_name == "ipointnet":
        from src.models.pointnet import IPointNetSegmentation

        model = IPointNetSegmentation(
            num_classes=cfg.num_classes,
            input_channels=cfg.num_channels,
            dropout=cfg.dropout_rate,
        ).to(device)
    else:
        raise ValueError(f"Model name {cfg.model_name} does not exist")

    # Loss + optimizer + scheduler
    loss_weights = torch.tensor(cfg.loss_weights, dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=loss_weights, gamma=focal_gamma, ignore_index=cfg.ignore_label)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    scheduler = None
    if getattr(cfg, "scheduler_type", None) == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.num_epochs,
            eta_min=getattr(cfg, "scheduler_min_lr", 1e-6),
        )

    def report_fn(**kwargs):
        tune.report(kwargs)

    train_model_segmentation(
        cfg,
        train_loader,
        val_loader,
        model,
        optimizer,
        criterion,
        scheduler,
        device,
        base_path,
        report_fn=report_fn,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=30)
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--gpus", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="runs/ray_tune")
    ap.add_argument(
        "--tune-scheduler-min-lr",
        action="store_true",
        help="Also tune scheduler_min_lr (only used for cosine scheduler).",
    )
    ap.add_argument(
        "--ray-tmp",
        type=str,
        default="raytmp",
        help="Short temp dir for Ray (helps avoid Windows MAX_PATH issues).",
    )
    args = ap.parse_args()

    # --- Make storage path unambiguous on Windows: use file:// URI ---
    out_dir = Path(args.out).expanduser()
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    storage_uri = out_dir.as_uri()  # file:///C:/.../runs/ray_tune
    print(f"[Ray Tune] storage_path = {storage_uri}")

    # --- Reduce long-path issues: force Ray temp dir to something short ---
    # Must be set BEFORE Ray auto-initializes (i.e., before tuner.fit()).
    ray_tmp = Path(args.ray_tmp).expanduser()
    if not ray_tmp.is_absolute():
        ray_tmp = Path.cwd() / ray_tmp
    ray_tmp = ray_tmp.resolve()
    ray_tmp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RAY_TMPDIR", str(ray_tmp))
    # (optional) if you still see long log paths, you can also shorten Python temp:
    # os.environ.setdefault("TMP", str(ray_tmp))
    # os.environ.setdefault("TEMP", str(ray_tmp))

    # Pre-compute all loss-weight variants from the actual training data so
    # every Ray trial gets consistent, data-driven options without re-scanning.
    cfg_for_weights, repo_root_for_weights = load_base_config()
    weight_options = precompute_class_weights(
        data_dir=Path(cfg_for_weights.model_data_path) / "train",
        ignore_label=cfg_for_weights.ignore_label,
        num_classes=getattr(cfg_for_weights, "num_classes", 5),
    )

    param_space = {
        "learning_rate": tune.loguniform(1e-5, 3e-3),
        "batch_size": tune.choice([8, 16, 24, 32]),
        "focal_gamma": tune.uniform(1.0, 3.0),
        "rare_class_boost": tune.uniform(1.0, 5.0),

        # Only used if cfg.dataset_augmentation=True
        "rotation_deg_max": tune.uniform(0.0, 20.0),
        "scale_min": tune.uniform(0.9, 1.0),
        "scale_max": tune.uniform(1.0, 1.1),

        # loss_weights: one pre-computed vector per weighting scheme
        # (sqrt_inv_freq / moderate / effective_num variants / uniform)
        "loss_weights": tune.choice(weight_options),

        "num_epochs": args.max_epochs,
        "seed": 42,
    }

    if args.tune_scheduler_min_lr:
        param_space["scheduler_min_lr"] = tune.loguniform(1e-7, 1e-4)

    scheduler = ASHAScheduler(
        time_attr="epoch",
        max_t=args.max_epochs,
        grace_period=max(3, args.max_epochs // 10),
        reduction_factor=2,
    )

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"cpu": args.cpus, "gpu": args.gpus}),
        tune_config=tune.TuneConfig(
            metric="val_miou",
            mode="max",
            num_samples=args.num_samples,
            scheduler=scheduler,
            # Keep trial directory names short to avoid Windows MAX_PATH (260 chars)
            trial_dirname_creator=lambda trial: f"trial_{trial.trial_id[:8]}",
        ),
        run_config=tune.RunConfig(
            name=f"dales_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            storage_path=storage_uri,
            verbose=1,
        ),
        param_space=param_space,
    )

    results = tuner.fit()

    # If all trials error / no metrics, avoid crashing here
    try:
        best = results.get_best_result(metric="val_miou", mode="max")
    except Exception as e:
        print("\nNo best trial found (likely all trials errored or no metrics reported).")
        print("Exception:", repr(e))
        print("Check trial error.txt files under:", out_dir)
        return

    print("\nBest trial:")
    print("  best val_miou:", best.metrics.get("val_miou"))
    print("  config:", best.config)
    print("  logdir:", best.path)


if __name__ == "__main__":
    main()