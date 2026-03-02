# src/tune_ray.py
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import os
import random
import numpy as np
import torch

from ray import tune
from ray.tune.schedulers import ASHAScheduler

from src.utils.config_parser import ConfigParser
from src.utils.dataset import DALESDataset
from torch.utils.data import DataLoader
from src.utils.trainer import train_model_segmentation
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


def load_base_config():
    """
    Loads existing YAML config via ConfigParser, and returns (cfg, repo_root_path).

    Run from repo root as:
        python -m src.tune_ray ...
    """
    base_path = Path(os.getcwd())
    if base_path.name == "src":
        base_path = base_path.parent

    config_parser = ConfigParser(
        default_config_path="config/default.yaml",
        parser=argparse.ArgumentParser(description="Ray Tune HPO for DALES Segmentation"),
    )
    cfg = config_parser.load()
    return cfg, base_path


def trainable(trial_cfg: dict):
    cfg, base_path = load_base_config()

    # Disable W&B + snapshots inside HPO trials
    cfg.wandb_enabled = False
    cfg.save_snapshots = False

    # Override hyperparameters
    cfg.learning_rate = float(trial_cfg["learning_rate"])
    cfg.batch_size = int(trial_cfg["batch_size"])
    cfg.num_epochs = int(trial_cfg["num_epochs"])

    focal_gamma = float(trial_cfg["focal_gamma"])
    rare_class_boost = float(trial_cfg["rare_class_boost"])

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

    # Datasets / loaders (same as main.py)
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

    # Optional loader args (can be set in config; defaults are safe)
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
            dropout=cfg.dropout_rate,  # segmentation dropout currently unused in your model
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

    # Ray reporting
    def report_fn(**kwargs):
        tune.report(**kwargs)

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
    args = ap.parse_args()

    param_space = {
        "learning_rate": tune.loguniform(1e-5, 3e-3),
        "batch_size": tune.choice([8, 16, 24, 32]),
        "focal_gamma": tune.uniform(1.0, 3.0),
        "rare_class_boost": tune.uniform(1.0, 5.0),

        # Only used if cfg.dataset_augmentation=True
        "rotation_deg_max": tune.uniform(0.0, 20.0),
        "scale_min": tune.uniform(0.9, 1.0),
        "scale_max": tune.uniform(1.0, 1.1),

        "num_epochs": args.max_epochs,
        "seed": 42,
    }

    if args.tune_scheduler_min_lr:
        param_space["scheduler_min_lr"] = tune.loguniform(1e-7, 1e-4)

    scheduler = ASHAScheduler(
        time_attr="epoch",
        metric="val_miou",
        mode="max",
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
        ),
        run_config=tune.RunConfig(
            name=f"dales_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            local_dir=args.out,
            verbose=1,
        ),
        param_space=param_space,
    )

    results = tuner.fit()
    best = results.get_best_result(metric="val_miou", mode="max")

    print("\nBest trial:")
    print("  best val_miou:", best.metrics.get("val_miou"))
    print("  config:", best.config)
    print("  logdir:", best.path)


if __name__ == "__main__":
    main()