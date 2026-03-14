import torch
import numpy as np
from tqdm import tqdm
import os
import wandb


# -----------------------------
# W&B table for visualization
# Logs only first, middle, and last epoch
# -----------------------------
wandb_pointcloud_table = wandb.Table(
    columns=["epoch", "GT", "Prediction", "Errors", "Uncertainty"]
)

# -----------------------------
# W&B table for point-cloud ↔ BEV image pairing
# -----------------------------
wandb_pc_image_table = wandb.Table(
    columns=[
        "epoch",
        "PointCloud",
        "BEV image density",
        "BEV image z_max",
        "BEV image z_mean",
        "BEV image z_range",
    ]
)


def compute_regularizationLoss(feature_tnet):
    # REGULARIZATION: force Tnet matrix to be orthogonal (TT^t = I)
    # i.e. allow transforming the sapce but without distorting it
    # The loss adds this term to be minimized: ||I-TT^t||
    # It is a training constrain --> no need to be included in validation
    TT = torch.bmm(feature_tnet, feature_tnet.transpose(2, 1))
    I = torch.eye(TT.shape[-1], device=TT.device).unsqueeze(0).expand(TT.shape[0], -1, -1)  # [64,64]->[1,64,64]->[B,64,64]
    reg_loss = torch.norm(I - TT) / TT.shape[0]  # make reg_loss batch invariant (dividing by batch_size)

    return reg_loss


# //////////////////////////////////////////////////////////////////////////////
#                     SEGMENTATION TRAINING FUNCTIONS
# //////////////////////////////////////////////////////////////////////////////

def compute_batch_intersection_and_union(labels, predictions, num_classes):

    inter_batch = torch.zeros(num_classes, device=predictions.device, dtype=torch.float32)
    union_batch = torch.zeros(num_classes, device=predictions.device, dtype=torch.float32)

    for c in range(num_classes):
        inter_batch[c] = ((predictions == c) & (labels == c)).sum()     # BOTH ARE c
        union_batch[c] = ((predictions == c) | (labels == c)).sum()     # EITHER ONE OR THE OTHER ARE c

    return inter_batch, union_batch


# -----------------------------
# W&B 3D point-cloud utilities
# -----------------------------
CLASS_COLOR_MAP = {
    -1: [128, 128, 128],  # ignored / unknown
    0: [0, 0, 255],       # Ground
    1: [0, 153, 0],       # Vegetation
    2: [255, 0, 0],       # Buildings
    3: [255, 217, 0],     # Vehicle
    4: [255, 128, 0],     # Utility
}


def _labels_to_rgb(labels_np: np.ndarray) -> np.ndarray:
    colors = np.zeros((labels_np.shape[0], 3), dtype=np.int32)
    for cls_id, rgb in CLASS_COLOR_MAP.items():
        mask = labels_np == cls_id
        colors[mask] = rgb
    return colors


def _make_colored_point_cloud(points_xyz: np.ndarray, labels_np: np.ndarray) -> np.ndarray:
    colors = _labels_to_rgb(labels_np)
    return np.concatenate([points_xyz.astype(np.float32), colors.astype(np.float32)], axis=1)


def _compute_entropy(log_probs_BCN: torch.Tensor) -> torch.Tensor:
    probs = torch.exp(log_probs_BCN)  # [B, C, N]
    entropy = -(probs * log_probs_BCN).sum(dim=1)  # [B, N]
    return entropy


def _entropy_to_rgb(entropy_np: np.ndarray) -> np.ndarray:
    e_min = entropy_np.min()
    e_max = entropy_np.max()
    norm = (entropy_np - e_min) / (e_max - e_min + 1e-8)

    r = (norm * 255).astype(np.int32)
    g = np.zeros_like(r)
    b = ((1.0 - norm) * 255).astype(np.int32)

    return np.stack([r, g, b], axis=1)


def _append_pointcloud_predictions_to_table(points_BNC, labels, predictions, log_probs_BCN, epoch: int):
    global wandb_pointcloud_table

    points_xyz = points_BNC[0, :, :3].detach().cpu().numpy()
    gt = labels[0].detach().cpu().numpy()
    pred = predictions[0].detach().cpu().numpy()
    entropy = _compute_entropy(log_probs_BCN)[0].detach().cpu().numpy()

    valid_mask = gt != -1
    points_xyz = points_xyz[valid_mask]
    gt = gt[valid_mask]
    pred = pred[valid_mask]
    entropy = entropy[valid_mask]

    gt_cloud = _make_colored_point_cloud(points_xyz, gt)
    pred_cloud = _make_colored_point_cloud(points_xyz, pred)

    err_mask = gt != pred
    err_points = points_xyz[err_mask]
    if len(err_points) > 0:
        err_colors = np.tile(np.array([[255, 0, 255]]), (len(err_points), 1))
        err_cloud = np.concatenate([err_points, err_colors], axis=1)
    else:
        err_cloud = np.zeros((0, 6), dtype=np.float32)

    entropy_colors = _entropy_to_rgb(entropy)
    entropy_cloud = np.concatenate([points_xyz, entropy_colors], axis=1)

    wandb_pointcloud_table.add_data(
        epoch + 1,
        wandb.Object3D(gt_cloud),
        wandb.Object3D(pred_cloud),
        wandb.Object3D(err_cloud),
        wandb.Object3D(entropy_cloud),
    )


# -----------------------------
# W&B point-cloud ↔ BEV image pairing utilities
# -----------------------------
def _normalize_image_for_wandb(img_np: np.ndarray) -> np.ndarray:
    img_np = img_np.astype(np.float32)
    img_np = np.nan_to_num(img_np, nan=0.0, posinf=0.0, neginf=0.0)

    vmin = np.min(img_np)
    vmax = np.max(img_np)

    if vmax - vmin < 1e-8:
        return np.zeros_like(img_np, dtype=np.uint8)

    img_norm = ((img_np - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    return img_norm


def _append_pointcloud_image_pair_to_table(points_BNC, labels, image, epoch: int):
    global wandb_pc_image_table

    points_xyz = points_BNC[0, :, :3].detach().cpu().numpy()
    gt = labels[0].detach().cpu().numpy()

    valid_mask = gt != -1
    points_xyz = points_xyz[valid_mask]
    gt = gt[valid_mask]

    # color point cloud using ground-truth classes
    pc_cloud = _make_colored_point_cloud(points_xyz, gt)

    if torch.is_tensor(image):
        img_np = image[0].detach().cpu().numpy()   # expected [C, H, W]
    else:
        img_np = image[0]

    if img_np.ndim != 3 or img_np.shape[0] < 4:
        raise ValueError(
            f"Expected BEV image with shape [C,H,W] and at least 4 channels, got {img_np.shape}"
        )

    density = _normalize_image_for_wandb(img_np[0])
    z_max = _normalize_image_for_wandb(img_np[1])
    z_mean = _normalize_image_for_wandb(img_np[2])
    z_range = _normalize_image_for_wandb(img_np[3])

    wandb_pc_image_table.add_data(
        epoch + 1,
        wandb.Object3D(pc_cloud),
        wandb.Image(density, caption=f"Epoch {epoch+1} density"),
        wandb.Image(z_max, caption=f"Epoch {epoch+1} z_max"),
        wandb.Image(z_mean, caption=f"Epoch {epoch+1} z_mean"),
        wandb.Image(z_range, caption=f"Epoch {epoch+1} z_range"),
    )


# ----------------------------------------------------
#    TRAINING EPOCH FUNCTION (SEGMENTATION)
# ----------------------------------------------------
def train_single_epoch_segmentation(config, train_loader, network, optimizer, criterion, scaler, use_image=False):

    device = next(network.parameters()).device  # guarantee that we are using the same device than the model
    network.train()                             # Activate the train=True flag inside the model

    loss_history = []
    nCorrect = 0
    nTotal = 0
    inter = torch.zeros(network.num_classes, device=device, dtype=torch.float32)
    union = torch.zeros(network.num_classes, device=device, dtype=torch.float32)
    for batch in tqdm(train_loader, desc="train epoch", position=1, leave=False):

        # Pointnet needs: [B, N, C]
        if(use_image):
            points_BNC, labels, image = batch                           # Points: [B, N, C] / labels: [B, N] / image [B, H, W]
        else:
            points_BNC, labels = batch                           # Points: [B, N, C] / labels: [B, N] / image [B, H, W]

        points_BNC = points_BNC.to(device)
        labels = labels.to(device)

        # Set network gradients to 0
        optimizer.zero_grad(set_to_none=True)

        # Forward pass + loss under mixed precision
        with torch.amp.autocast(device.type):
            # Forward points and image through the network
            if use_image:
                image = image.permute(0,3,1,2)  #change [B, H, W, C] -> [B. C. H. W]
                image = image.to(device)
                output = network(points_BNC, image)
            else:
                output = network(points_BNC)

            # Handle output depending on what model returns
            if isinstance(output, tuple):
                feature_tnet, log_probs_BCN = output
                reg_loss = compute_regularizationLoss(feature_tnet)
            else:
                log_probs_BCN = output
                reg_loss = torch.tensor(0.0, device=device)

            # Compute loss: NLLLoss(ignore_index=-1)
            # NLLLoss expects class dimension at dim=1, network returns [B, num_classes, N] --> HAPPY!
            loss = criterion(log_probs_BCN, labels) + 0.001 * reg_loss

        loss_history.append(loss.item())

        # Mixed precision backward and optimizer step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ----------- COMPUTE METRICS -------------
        # Compute predictions
        predictions = log_probs_BCN.argmax(dim=1)
        # Identify valid labels (-1 is not valid)
        id_valid = labels != config.ignore_label
        valid_predictions = predictions[id_valid]
        valid_labels = labels[id_valid]
        # Accuracy
        batch_correct = (valid_predictions == valid_labels).sum().item()    # num correct (valid) per batch
        nCorrect = nCorrect + batch_correct                                 # num correct (valid) per epoch
        nTotal = nTotal + id_valid.sum().item()                             # num total (valid) per epoch
        # Update intersection and union
        inter_batch, union_batch = compute_batch_intersection_and_union(valid_labels, valid_predictions, network.num_classes)
        inter += inter_batch
        union += union_batch
        # ------------------------------------------

    assert nTotal > 0, "No valid points in epoch (all labels are -1)."
    assert (union > 0).any(), "Not a single class present for IoU in epoch."
    # Average across all batches
    train_loss_epoch = np.mean(loss_history)
    train_acc_epoch = nCorrect / nTotal
    # Compute IoU per class and mean
    id_present = union > 0                                                    # id of classes that are present
    iou_class_epoch = torch.zeros(network.num_classes, device=device, dtype=torch.float32)
    iou_class_epoch[id_present] = inter[id_present] / union[id_present]       # iou of each class, per epoch (could be returned)
    train_miou_epoch = iou_class_epoch[id_present].mean().item()              # mean iou over classes, per epoch

    return train_loss_epoch, train_acc_epoch, train_miou_epoch, iou_class_epoch
# ----------------------------------------------------


# ----------------------------------------------------
#    TESTING EPOCH FUNCTION (SEGMENTATION)
# ----------------------------------------------------
def eval_single_epoch_segmentation(config, data_loader, network, criterion, use_image=False, epoch=None):

    device = next(network.parameters()).device  # guarantee that we are using the same device than the model

    with torch.no_grad():
        network.eval()                      # Dectivate the train=True flag inside the model

        loss_history = []
        nCorrect = 0
        nTotal = 0
        inter = torch.zeros(network.num_classes, device=device, dtype=torch.float32)
        union = torch.zeros(network.num_classes, device=device, dtype=torch.float32)

        for batch_idx, batch in enumerate(tqdm(data_loader, desc="val epoch", position=1, leave=False)):

            # Pointnet needs: [B, N, C]
            if(use_image):
                points_BNC, labels, image = batch  # Points: [B, N, C] / labels: [B, N] / image [B, H, W]
            else:
                points_BNC, labels = batch  # Points: [B, N, C] / labels: [B, N] / image [B, H, W]
            
            points_BNC = points_BNC.to(device)
            labels = labels.to(device)

            # Forward pass + loss under mixed precision
            with torch.amp.autocast(device.type):
                # Forward points through the network
                if use_image:
                    image = image.permute(0,3,1,2) #change [B, H, W, C] -> [B. C. H. W]
                    image = image.to(device)
                    output = network(points_BNC, image)
                else:
                    output = network(points_BNC)

                # Handle output depending on what model returns
                if isinstance(output, tuple):
                    feature_tnet, log_probs_BCN = output
                else:
                    log_probs_BCN = output

                # Compute loss: NLLLoss(ignore_index=-1)
                # NLLLoss expects class dimension at dim=1, network returns [B, num_classes, N] --> HAPPY!
                loss = criterion(log_probs_BCN, labels)

            loss_history.append(loss.item())

            # ----------- COMPUTE METRICS -------------
            # Compute predictions
            predictions = log_probs_BCN.argmax(dim=1)
            # Identify valid labels (-1 is not valid)
            id_valid = labels != config.ignore_label
            valid_predictions = predictions[id_valid]
            valid_labels = labels[id_valid]
            # Accuracy
            batch_correct = (valid_predictions == valid_labels).sum().item()    # num correct (valid) per batch
            nCorrect = nCorrect + batch_correct                                 # num correct (valid) per epoch
            nTotal = nTotal + id_valid.sum().item()                             # num total (valid) per epoch
            # Update intersection and union
            inter_batch, union_batch = compute_batch_intersection_and_union(valid_labels, valid_predictions, network.num_classes)
            inter += inter_batch
            union += union_batch

            # Log visualization only for first, middle, and last epoch, using first validation batch only
            if batch_idx == 0 and epoch is not None and epoch in {0, config.num_epochs // 2, config.num_epochs - 1}:
                _append_pointcloud_predictions_to_table(points_BNC, labels, predictions, log_probs_BCN, epoch)
                if(use_image):
                    _append_pointcloud_image_pair_to_table(points_BNC, labels, image, epoch)
            # ------------------------------------------

        assert nTotal > 0, "No valid points in epoch (all labels are -1)."
        assert (union > 0).any(), "Not a single class present for IoU in epoch."
        # Average across all batches
        eval_loss_epoch = np.mean(loss_history)
        eval_acc_epoch = nCorrect / nTotal
        # Compute IoU per class and mean
        id_present = union > 0                                                    # id of classes that are present
        iou_class_epoch = torch.zeros(network.num_classes, device=device, dtype=torch.float32)
        iou_class_epoch[id_present] = inter[id_present] / union[id_present]       # iou of each class, per epoch (could be returned)
        eval_miou_epoch = iou_class_epoch[id_present].mean().item()               # mean iou over classes, per epoch

    return eval_loss_epoch, eval_acc_epoch, eval_miou_epoch, iou_class_epoch
# ----------------------------------------------------


# ----------------------------------------------------
#    TRAINING LOOP (iterate on epochs)
# ----------------------------------------------------
def train_model_segmentation(config, train_loader, val_loader, network, optimizer, criterion, scheduler, device, base_path):

    global wandb_pointcloud_table
    global wandb_pc_image_table

    use_image = False
    if config.model_name == "ipointnet":
        use_image = True

    # Reset W&B visualization table for each run
    wandb_pointcloud_table = wandb.Table(
        columns=["epoch", "GT", "Prediction", "Errors", "Uncertainty"]
    )

    # Reset point-cloud ↔ BEV image table for each run
    wandb_pc_image_table = wandb.Table(
        columns=[
            "epoch",
            "PointCloud",
            "BEV image density",
            "BEV image z_max",
            "BEV image z_mean",
            "BEV image z_range",
        ]
    )

    metrics = {
        "train_loss": [],
        "train_acc": [],
        "train_miou": [],
        "val_loss": [],
        "val_acc": [],
        "val_miou": []
    }

    # Mixed precision scaler
    scaler = torch.amp.GradScaler(device.type)

    for epoch in tqdm(range(config.num_epochs), desc="Looping on epochs", position=0):
        train_loss_epoch, train_acc_epoch, train_miou_epoch, train_iou_class_epoch = train_single_epoch_segmentation(config, train_loader, network, optimizer, criterion, scaler, use_image)

        val_loss_epoch, val_acc_epoch, val_miou_epoch, val_iou_class_epoch = eval_single_epoch_segmentation(
            config, val_loader, network, criterion, use_image, epoch=epoch
        )

        metrics["train_loss"].append(train_loss_epoch)
        metrics["train_acc"].append(train_acc_epoch)
        metrics["train_miou"].append(train_miou_epoch)
        metrics["val_loss"].append(val_loss_epoch)
        metrics["val_acc"].append(val_acc_epoch)
        metrics["val_miou"].append(val_miou_epoch)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        tqdm.write(f"Epoch: {epoch+1}/{config.num_epochs}"
            f" | loss (train/val) = {train_loss_epoch:.3f}/{val_loss_epoch:.3f}"
            f" | acc (train/val) = {train_acc_epoch:.3f}/{val_acc_epoch:.3f}"
            f" | miou (train/val) = {train_miou_epoch:.3f}/{val_miou_epoch:.3f}"
            f" | lr = {current_lr:.6f}")

        # Step the learning rate scheduler
        scheduler.step()

        if(epoch % config.snap_interval == 0):
            checkpoint = {
                "model_state_dict": network.state_dict(),
                "config": config,   # save full configuration
            }
            path_to_save = os.path.join(base_path, "snapshots", config.test_name)
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            path_to_save = os.path.join(path_to_save, f"pointnet_{epoch}_epochs.pt")
            torch.save(checkpoint, path_to_save)

        # Class names for DALES dataset
        class_names = ["Ground", "Vegetation", "Buildings", "Vehicle", "Utility"]

        # Build wandb log dict with per-class IoU
        log_dict = {
            "Loss/Train": train_loss_epoch,
            "Loss/Validation": val_loss_epoch,
            "Accuracy/Train": train_acc_epoch,
            "Accuracy/Validation": val_acc_epoch,
            "mIoU/Train": train_miou_epoch,
            "mIoU/Validation": val_miou_epoch,
            "Learning_Rate": current_lr,
        }

        # Add per-class IoU metrics
        for i, class_name in enumerate(class_names):
            log_dict[f"IoU_Class/{class_name}/Train"] = train_iou_class_epoch[i].item()
            log_dict[f"IoU_Class/{class_name}/Validation"] = val_iou_class_epoch[i].item()

        wandb.log(log_dict, step=epoch+1)

        # show a pointcloud from training with the transformation
        """
        for pc, label, img in train_loader:
        pcs = pc[:1,:,:3]
        writer.add_mesh("pointcloud", vertices=pcs, global_step=0)
        break
        """

    # Log the full accumulated visualization tables once at the end
    wandb.log({"Segmentation_Visualization": wandb_pointcloud_table})
    wandb.log({"Point Cloud & Images": wandb_pc_image_table})

    return metrics
