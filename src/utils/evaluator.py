import torch
import os
import wandb
from utils.trainer import eval_single_epoch_segmentation
from tqdm import tqdm

# ----------------------------------------------------
#    TESTING LOOP (iterate on epochs)
# ----------------------------------------------------
def test_model_segmentation(config, test_loader, network, criterion, device, base_path):

    use_image = False
    if config.model_name == "ipointnet":
        use_image = True

    metrics = {"test_loss": [],   
               "test_acc": [],   
               "test_miou": []}
    
    # Evaluate on test set (single pass, not multiple epochs)
    test_loss_epoch, test_acc_epoch, test_miou_epoch, test_iou_class_epoch = eval_single_epoch_segmentation(config, test_loader, network, criterion, use_image)
    
    metrics["test_loss"].append(test_loss_epoch)
    metrics["test_acc"].append(test_acc_epoch)
    metrics["test_miou"].append(test_miou_epoch)
    
    print(f"Test Results:"
        f" | loss = {test_loss_epoch:.3f}"
        f" | acc = {test_acc_epoch:.3f}"
        f" | miou = {test_miou_epoch:.3f}")
    
    # Class names for DALES dataset
    class_names = ["Ground", "Vegetation", "Buildings", "Vehicle", "Utility"]

    # Build wandb log dict with per-class IoU
    log_dict = {
        "Loss/Test": test_loss_epoch,
        "Accuracy/Test": test_acc_epoch,
        "mIoU/Test": test_miou_epoch,
    }
    
    # Add per-class IoU metrics
    for i, class_name in enumerate(class_names):
        log_dict[f"IoU_Class/{class_name}/Test"] = test_iou_class_epoch[i].item()
    
    wandb.log(log_dict)
    return metrics
