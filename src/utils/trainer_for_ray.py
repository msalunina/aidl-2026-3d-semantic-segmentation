import torch
import numpy as np
from tqdm import tqdm
import os

# Make wandb optional (so Ray Tune runs don't require it installed)
try:
    import wandb  # type: ignore
except ImportError:
    wandb = None


def compute_regularizationLoss(feature_tnet):
    # REGULARIZATION: force Tnet matrix to be orthogonal (TT^t = I)
    # i.e. allow transforming the space but without distorting it
    # The loss adds this term to be minimized: ||I-TT^t||
    # It is a training constraint --> no need to be included in validation
    TT = torch.bmm(feature_tnet, feature_tnet.transpose(2, 1))
    I = torch.eye(TT.shape[-1], device=TT.device).unsqueeze(0).expand(TT.shape[0], -1, -1)
    reg_loss = torch.norm(I - TT) / TT.shape[0]  # batch invariant
    return reg_loss


# //////////////////////////////////////////////////////////////////////////////
#                     SEGMENTATION TRAINING FUNCTIONS
# //////////////////////////////////////////////////////////////////////////////

def compute_batch_intersection_and_union(labels, predictions, num_classes):
    inter_batch = torch.zeros(num_classes, device=predictions.device, dtype=torch.float32)
    union_batch = torch.zeros(num_classes, device=predictions.device, dtype=torch.float32)

    for c in range(num_classes):
        inter_batch[c] = ((predictions == c) & (labels == c)).sum()  # BOTH ARE c
        union_batch[c] = ((predictions == c) | (labels == c)).sum()  # EITHER ONE OR THE OTHER ARE c

    return inter_batch, union_batch


# ----------------------------------------------------
#    TRAINING EPOCH FUNCTION (SEGMENTATION)
# ----------------------------------------------------
def train_single_epoch_segmentation(config, train_loader, network, optimizer, criterion, use_image=False):
    device = next(network.parameters()).device
    network.train()

    loss_history = []
    nCorrect = 0
    nTotal = 0
    inter = torch.zeros(network.num_classes, device=device, dtype=torch.float32)
    union = torch.zeros(network.num_classes, device=device, dtype=torch.float32)

    for batch in tqdm(train_loader, desc="train epoch", position=1, leave=False):
        # PointNet needs: [B, N, C]
        if use_image:
            points_BNC, labels, image = batch
            image = image.unsqueeze(dim=1)  # [B, C, H, W]
            image = image.to(device)
        else:
            points_BNC, labels = batch

        points_BNC = points_BNC.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward
        if use_image:
            feature_tnet, log_probs_BCN = network(points_BNC, image)
        else:
            feature_tnet, log_probs_BCN = network(points_BNC)

        # Loss (criterion expects [B, C, N] and labels [B, N])
        reg_loss = compute_regularizationLoss(feature_tnet)
        loss = criterion(log_probs_BCN, labels) + 0.001 * reg_loss
        loss_history.append(loss.item())

        loss.backward()
        optimizer.step()

        # ----------- METRICS -------------
        predictions = log_probs_BCN.argmax(dim=1)
        id_valid = labels != -1  # config.ignore_label
        valid_predictions = predictions[id_valid]
        valid_labels = labels[id_valid]

        batch_correct = (valid_predictions == valid_labels).sum().item()
        nCorrect += batch_correct
        nTotal += id_valid.sum().item()

        inter_batch, union_batch = compute_batch_intersection_and_union(
            valid_labels, valid_predictions, network.num_classes
        )
        inter += inter_batch
        union += union_batch
        # ---------------------------------

    assert nTotal > 0, "No valid points in epoch (all labels are -1)."
    assert (union > 0).any(), "Not a single class present for IoU in epoch."

    train_loss_epoch = float(np.mean(loss_history))
    train_acc_epoch = float(nCorrect / nTotal)

    id_present = union > 0
    iou_class_epoch = torch.zeros(network.num_classes, device=device, dtype=torch.float32)
    iou_class_epoch[id_present] = inter[id_present] / union[id_present]
    train_miou_epoch = float(iou_class_epoch[id_present].mean().item())

    return train_loss_epoch, train_acc_epoch, train_miou_epoch, iou_class_epoch


# ----------------------------------------------------
#    EVAL EPOCH FUNCTION (SEGMENTATION)
# ----------------------------------------------------
def eval_single_epoch_segmentation(config, data_loader, network, criterion, use_image=False):
    device = next(network.parameters()).device

    with torch.no_grad():
        network.eval()

        loss_history = []
        nCorrect = 0
        nTotal = 0
        inter = torch.zeros(network.num_classes, device=device, dtype=torch.float32)
        union = torch.zeros(network.num_classes, device=device, dtype=torch.float32)

        for batch in tqdm(data_loader, desc="val epoch", position=1, leave=False):
            if use_image:
                points_BNC, labels, image = batch
                image = image.unsqueeze(dim=1)
                image = image.to(device)
            else:
                points_BNC, labels = batch

            points_BNC = points_BNC.to(device)
            labels = labels.to(device)

            if use_image:
                feature_tnet, log_probs_BCN = network(points_BNC, image)
            else:
                feature_tnet, log_probs_BCN = network(points_BNC)

            loss = criterion(log_probs_BCN, labels)
            loss_history.append(loss.item())

            predictions = log_probs_BCN.argmax(dim=1)
            id_valid = labels != -1  # config.ignore_label
            valid_predictions = predictions[id_valid]
            valid_labels = labels[id_valid]

            batch_correct = (valid_predictions == valid_labels).sum().item()
            nCorrect += batch_correct
            nTotal += id_valid.sum().item()

            inter_batch, union_batch = compute_batch_intersection_and_union(
                valid_labels, valid_predictions, network.num_classes
            )
            inter += inter_batch
            union += union_batch

        assert nTotal > 0, "No valid points in epoch (all labels are -1)."
        assert (union > 0).any(), "Not a single class present for IoU in epoch."

        eval_loss_epoch = float(np.mean(loss_history))
        eval_acc_epoch = float(nCorrect / nTotal)

        id_present = union > 0
        iou_class_epoch = torch.zeros(network.num_classes, device=device, dtype=torch.float32)
        iou_class_epoch[id_present] = inter[id_present] / union[id_present]
        eval_miou_epoch = float(iou_class_epoch[id_present].mean().item())

    return eval_loss_epoch, eval_acc_epoch, eval_miou_epoch, iou_class_epoch


# ----------------------------------------------------
#    TRAINING LOOP (iterate on epochs) + Ray Tune support
# ----------------------------------------------------
def train_model_segmentation(
    config,
    train_loader,
    val_loader,
    network,
    optimizer,
    criterion,
    scheduler,
    device,
    base_path,
    report_fn=None,  # callback for Ray Tune (e.g., tune.report)
):
    use_image = (config.model_name == "ipointnet")

    # Optional flags (default to current behavior)
    wandb_enabled = getattr(config, "wandb_enabled", True)
    save_snapshots = getattr(config, "save_snapshots", True)
    snap_interval = getattr(config, "snap_interval", 1)

    metrics = {
        "train_loss": [],
        "train_acc": [],
        "train_miou": [],
        "val_loss": [],
        "val_acc": [],
        "val_miou": [],
    }

    for epoch in tqdm(range(config.num_epochs), desc="Looping on epochs", position=0):
        train_loss_epoch, train_acc_epoch, train_miou_epoch, train_iou_class_epoch = train_single_epoch_segmentation(
            config, train_loader, network, optimizer, criterion, use_image
        )

        val_loss_epoch, val_acc_epoch, val_miou_epoch, val_iou_class_epoch = eval_single_epoch_segmentation(
            config, val_loader, network, criterion, use_image
        )

        metrics["train_loss"].append(train_loss_epoch)
        metrics["train_acc"].append(train_acc_epoch)
        metrics["train_miou"].append(train_miou_epoch)
        metrics["val_loss"].append(val_loss_epoch)
        metrics["val_acc"].append(val_acc_epoch)
        metrics["val_miou"].append(val_miou_epoch)

        current_lr = optimizer.param_groups[0]["lr"]

        tqdm.write(
            f"Epoch: {epoch+1}/{config.num_epochs}"
            f" | loss (train/val) = {train_loss_epoch:.3f}/{val_loss_epoch:.3f}"
            f" | acc (train/val) = {train_acc_epoch:.3f}/{val_acc_epoch:.3f}"
            f" | miou (train/val) = {train_miou_epoch:.3f}/{val_miou_epoch:.3f}"
            f" | lr = {current_lr:.6f}"
        )

        # ✅ Ray Tune reporting hook
        # Use 1-based epoch so ASHA max_t aligns with num_epochs.
        if report_fn is not None:
            report_fn(
                epoch=epoch + 1,
                train_loss=float(train_loss_epoch),
                val_loss=float(val_loss_epoch),
                train_acc=float(train_acc_epoch),
                val_acc=float(val_acc_epoch),
                train_miou=float(train_miou_epoch),
                val_miou=float(val_miou_epoch),
                lr=float(current_lr),
            )

        # Step scheduler if present
        if scheduler is not None:
            scheduler.step()

        # Snapshots (optional; disable during Ray trials)
        if save_snapshots and (epoch % snap_interval == 0):
            checkpoint = {"model_state_dict": network.state_dict()}
            path_to_save = os.path.join(base_path, "snapshots", config.test_name)
            os.makedirs(path_to_save, exist_ok=True)
            torch.save(checkpoint, os.path.join(path_to_save, f"pointnet_{epoch}_epochs.pt"))

        # W&B logging (optional; disable during Ray trials)
        if wandb_enabled and wandb is not None:
            class_names = ["Ground", "Vegetation", "Buildings", "Vehicle", "Utility"]

            log_dict = {
                "Loss/Train": train_loss_epoch,
                "Loss/Validation": val_loss_epoch,
                "Accuracy/Train": train_acc_epoch,
                "Accuracy/Validation": val_acc_epoch,
                "mIoU/Train": train_miou_epoch,
                "mIoU/Validation": val_miou_epoch,
                "Learning_Rate": current_lr,
            }

            for i, class_name in enumerate(class_names):
                log_dict[f"IoU_Class/{class_name}/Train"] = train_iou_class_epoch[i].item()
                log_dict[f"IoU_Class/{class_name}/Validation"] = val_iou_class_epoch[i].item()

            wandb.log(log_dict, step=epoch + 1)

    return metrics