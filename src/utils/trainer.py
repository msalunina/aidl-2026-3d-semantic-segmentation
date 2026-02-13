import torch
import numpy as np
from tqdm import tqdm
import os


def compute_regularizationLoss(feature_tnet):
    # REGULARIZATION: force Tnet matrix to be orthogonal (TT^t = I)
    # i.e. allow transforming the sapce but without distorting it
    # The loss adds this term to be minimized: ||I-TT^t||
    # It is a training constrain --> no need to be included in validation
    TT = torch.bmm(feature_tnet, feature_tnet.transpose(2, 1))
    I = torch.eye(TT.shape[-1], device=TT.device).unsqueeze(0).expand(TT.shape[0], -1, -1) # [64,64]->[1,64,64]->[B,64,64]
    reg_loss = torch.norm(I - TT) / TT.shape[0]                 # make reg_loss batch invariant (dividing by batch_size)

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


# ----------------------------------------------------
#    TRAINING EPOCH FUNCTION (SEGMENTATION)
# ----------------------------------------------------
def train_single_epoch_segmentation(config, train_loader, network, optimizer, criterion):
    
    device = next(network.parameters()).device  # guarantee that we are using the same device than the model
    network.train()                             # Activate the train=True flag inside the model

    loss_history = []
    nCorrect = 0
    nTotal = 0
    inter = torch.zeros(network.num_classes, device=device, dtype=torch.float32)
    union = torch.zeros(network.num_classes, device=device, dtype=torch.float32)
    for batch in tqdm(train_loader, desc="train epoch", position=1, leave=False):

        # Pointnet needs: [B, N, C] 
        points_BNC, labels = batch                                   # Points: [B, N, C] / labels: [B, N]
        points_BNC = points_BNC.to(device)
        labels = labels.to(device) 

        # Set network gradients to 0
        optimizer.zero_grad()  

        # Forward points through the network                                    
        feature_tnet, log_probs_BCN = network(points_BNC)             
        
        # Compute loss: NLLLoss(ignore_index=-1) 
        # NLLLoss expects class dimension at dim=1, network returns [B, num_classes, N] --> HAPPY!
        reg_loss = compute_regularizationLoss(feature_tnet)          # compute regularization term
        loss = criterion(log_probs_BCN, labels) + 0.001 * reg_loss   # add it to the loss  
        loss_history.append(loss.item())         
        
        loss.backward()                                           
        optimizer.step()    

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
    id_present = union>0                                                    # id of classes that are present
    iou_class_epoch = torch.zeros(network.num_classes, device=device, dtype=torch.float32)
    iou_class_epoch[id_present] = inter[id_present] / union[id_present]     # iou of each class, per epoch (could be returned)
    train_miou_epoch = iou_class_epoch[id_present].mean().item()            # mean iou over classes, per epoch
    
    return train_loss_epoch, train_acc_epoch, train_miou_epoch
# ----------------------------------------------------



# ----------------------------------------------------
#    TESTING EPOCH FUNCTION (SEGMENTATION)
# ----------------------------------------------------
def eval_single_epoch_segmentation(config, data_loader, network, criterion):

    device = next(network.parameters()).device  # guarantee that we are using the same device than the model

    with torch.no_grad():
        network.eval()                      # Dectivate the train=True flag inside the model

        loss_history = []
        nCorrect = 0
        nTotal = 0
        inter = torch.zeros(network.num_classes, device=device, dtype=torch.float32)
        union = torch.zeros(network.num_classes, device=device, dtype=torch.float32)
        for batch in tqdm(data_loader, desc="val epoch", position=1, leave=False):
            
            # Pointnet needs: [B, N, C] 
            points_BNC, labels = batch                              # Points: [B, N, C] / labels: [B, N]  
            points_BNC = points_BNC.to(device)
            labels = labels.to(device) 

            # Forward points through the network 
            _, log_probs_BCN = network(points_BNC)                  

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
            # ------------------------------------------
        
        assert nTotal > 0, "No valid points in epoch (all labels are -1)."
        assert (union > 0).any(), "Not a single class present for IoU in epoch."
        # Average across all batches    
        eval_loss_epoch = np.mean(loss_history) 
        eval_acc_epoch = nCorrect / nTotal
        # Compute IoU per class and mean
        id_present = union>0                                                    # id of classes that are present
        iou_class_epoch = torch.zeros(network.num_classes, device=device, dtype=torch.float32)
        iou_class_epoch[id_present] = inter[id_present] / union[id_present]     # iou of each class, per epoch (could be returned)
        eval_miou_epoch = iou_class_epoch[id_present].mean().item()             # mean iou over classes, per epoch
    
    return eval_loss_epoch, eval_acc_epoch, eval_miou_epoch
# ----------------------------------------------------



# ----------------------------------------------------
#    TRAINING LOOP (iterate on epochs)
# ----------------------------------------------------
def train_model_segmentation(config, train_loader, val_loader, network, optimizer, criterion, writer, device, base_path):

    metrics = {
        "train_loss": [],
        "train_acc": [], 
        "train_miou": [],
        "val_loss": [],   
        "val_acc": [],   
        "val_miou": []
        }

    for epoch in tqdm(range(config.num_epochs), desc="Looping on epochs", position=0):
        train_loss_epoch, train_acc_epoch, train_miou_epoch = train_single_epoch_segmentation(config, train_loader, network, optimizer, criterion)

        val_loss_epoch, val_acc_epoch, val_miou_epoch = eval_single_epoch_segmentation(config, val_loader, network, criterion)
        
        metrics["train_loss"].append(train_loss_epoch)
        metrics["train_acc"].append(train_acc_epoch)
        metrics["train_miou"].append(train_miou_epoch)
        metrics["val_loss"].append(val_loss_epoch)
        metrics["val_acc"].append(val_acc_epoch)
        metrics["val_miou"].append(val_miou_epoch)

        tqdm.write(f"Epoch: {epoch+1}/{config.num_epochs}"
            f" | loss (train/val) = {train_loss_epoch:.3f}/{val_loss_epoch:.3f}"
            f" | acc (train/val) = {train_acc_epoch:.3f}/{val_acc_epoch:.3f}"
            f" | miou (train/val) = {train_miou_epoch:.3f}/{val_miou_epoch:.3f}")
        
        if(epoch % config.snap_interval == 0):
            checkpoint = {
                "model_state_dict": network.cpu().state_dict(),
            }
            path_to_save = os.path.join(base_path, "snapshots", config.test_name)
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            path_to_save = os.path.join(path_to_save,f"pointnet_{epoch}_epochs.pt")
            torch.save(checkpoint, path_to_save)
            network.to(device)
        
        #add if to log tensorbard?
        test_name = f"{config.test_name}_{config.num_epochs}_epochs" #get this name from the config?
        writer.add_scalar(tag=f"{test_name}_Train/Loss", scalar_value=train_loss_epoch, global_step=epoch)
        writer.add_scalar(tag=f"{test_name}_Train/Accuracy", scalar_value=train_acc_epoch, global_step=epoch)
        writer.add_scalar(tag=f"{test_name}_Train/mIoU", scalar_value=train_miou_epoch, global_step=epoch)
        writer.add_scalar(tag=f"{test_name}_Validation/Loss", scalar_value=val_loss_epoch, global_step=epoch)
        writer.add_scalar(tag=f"{test_name}_Validation/Accuracy", scalar_value=val_acc_epoch, global_step=epoch)
        writer.add_scalar(tag=f"{test_name}_Validation/mIoU", scalar_value=val_miou_epoch, global_step=epoch)

    return metrics
