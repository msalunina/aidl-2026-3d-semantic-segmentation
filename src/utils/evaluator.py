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
    
    for epoch in tqdm(range(config.num_epochs), desc="Looping on epochs", position=0):

        test_loss_epoch, test_acc_epoch, test_miou_epoch = eval_single_epoch_segmentation(config, test_loader, network, criterion, use_image)
        
        metrics["test_loss"].append(test_loss_epoch)
        metrics["test_acc"].append(test_acc_epoch)
        metrics["test_miou"].append(test_miou_epoch)
        
        tqdm.write(f"Epoch: {epoch+1}/{config.num_epochs}"
            f" | loss (test) = {test_loss_epoch:.3f}"
            f" | acc (test) = {test_acc_epoch:.3f}"
            f" | miou (tesr) = {test_miou_epoch:.3f}")
        
        wandb.log({
            "Loss/Test": test_loss_epoch,
            "Accuracy/Test": test_acc_epoch,
            "mIoU/Test": test_miou_epoch,
        }, step=epoch+1)
        
        #show a pointcloud from training with the transformation
        """
        for pc, label, img in train_loader:
        pcs = pc[:1,:,:3]
        writer.add_mesh("pointcloud", vertices=pcs, global_step=0)
        break
        """
    return metrics
