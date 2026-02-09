import torch
import numpy as np
import random
from torch_geometric.utils import to_dense_batch



# ----------------------------------------------------
#                      SET SEED
# ----------------------------------------------------
def set_seed(seed=42):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)



# //////////////////////////////////////////////////////////////////////////////
#                      CLASSIFICATION TRAINING FUNCTIONS
# //////////////////////////////////////////////////////////////////////////////
# ----------------------------------------------------
#           UNPACK CLASSIFICATION BATCH
# ----------------------------------------------------
def unpack_batch(batch):
    """
    Supports both:
    - PyG ModelNet batches (Data object with .pos and .y)
    - ShapeNet batches (tuple: points, object_class, seg_labels, num_seg_classes)
    Returns:
      x: [B, N, 3]
      y: [B]
    """
    # ---------- PyG / ModelNet ----------
    if hasattr(batch, "pos") and hasattr(batch, "y"):
        # Pointnet needs: [object, nPoints, coordinades] 
        # i.e. [32 object, 1024 points, 3 coordinates]: [batch_size, nPoints, 3]
        # Since batch.batch is [32x1024, 3] we have to split it into individual object points.
        # It could be done manually but "to_dense_batch" does that.
        # "to_dense_batch" will also add padding if not all objects have same number of points. In our case they have.
        # mask is a [batch_size, nPoints] boolean saying if an entry is actually a real point or padding
        points_BNC, _ = to_dense_batch(batch.pos, batch.batch)  
        labels_B = batch.y                                                 
    
        return points_BNC, labels_B

    # ---------- ShapeNet ----------
    elif isinstance(batch, (tuple, list)):
        
        points, object_class, seg_labels, global_labels = batch

        # Even though DataLoader will turn into torch tensors, 
        # I do this to guarantee I can use this function if I bypass DataLoader and use dataset directly
        points_BNC = torch.as_tensor(points, dtype=torch.float32)            
        labels_B = torch.as_tensor(object_class, dtype=torch.long)
        
        points_BNC = points_BNC.transpose(-2,-1)

        return points_BNC, labels_B

    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")


# ----------------------------------------------------
#    TRAINING EPOCH FUNCTION (iterate on data_loader)
# ----------------------------------------------------
def train_single_epoch(train_loader, network, optimizer, criterion):
    
    device = next(network.parameters()).device  # guarantee that we are using the same device than the model
    network.train()                             # Activate the train=True flag inside the model

    loss_history = []
    nCorrect = 0
    nTotal = 0
    for batch in train_loader:
        
        # Pointnet needs: [batch, nPoints, coordinates] 
        points_BNC, labels_B = unpack_batch(batch)
        points_BNC = points_BNC.to(device)
        labels_B = labels_B.to(device) 

        # forward batch and loss
        optimizer.zero_grad()                  # Set network gradients to 0
        log_probs, _, _, feature_tnet, _ = network(points_BNC)    # Forward batch through the network  
        reg_loss = compute_regularizationLoss(feature_tnet)
        loss = criterion(log_probs, labels_B) + 0.001 * reg_loss       # Compute loss: NLLLoss   
        loss_history.append(loss.item())         

        loss.backward()                                                 # Compute backpropagation
        optimizer.step()    
        
        # Compute metrics
        predictions = log_probs.argmax(dim=1)
        batch_correct = (predictions == labels_B).sum().item()          # .item() brings one single scalar to CPU
        nCorrect = nCorrect + batch_correct
        nTotal = nTotal + len(labels_B)

    # Average across all batches    
    train_loss_epoch = np.mean(loss_history) 
    train_acc_epoch = nCorrect / nTotal
    
    return train_loss_epoch, train_acc_epoch
# ----------------------------------------------------


# ----------------------------------------------------
#    TESTING EPOCH FUNCTION (iterate on data_loader)
# ----------------------------------------------------
def eval_single_epoch(data_loader, network, criterion):

    device = next(network.parameters()).device  # guarantee that we are using the same device than the model

    with torch.no_grad():
        network.eval()                      # Dectivate the train=True flag inside the model

        loss_history = []
        nCorrect = 0
        nTotal = 0
        for batch in data_loader:
            # Pointnet needs: [batch, nPoints, coordinates] 
            points_BNC, labels_B = unpack_batch(batch) 
            points_BNC = points_BNC.to(device)
            labels_B = labels_B.to(device) 
            
            # forward batch and loss
            log_probs, _, _, _, _ = network(points_BNC)         # Forward batch through the network
            loss = criterion(log_probs, labels_B)                  # Compute loss
            loss_history.append(loss.item())         
            
            # Compute metrics
            predictions = log_probs.argmax(dim=1)
            batch_correct = (predictions == labels_B).sum().item()  # .item() brings one single scalar to CPU
            nCorrect = nCorrect + batch_correct
            nTotal = nTotal + len(labels_B)     # classification: 1 prediction per sample
        
        # Average across all batches 
        eval_loss_epoch = np.mean(loss_history)       
        eval_acc_epoch = nCorrect / nTotal
    
    return eval_loss_epoch, eval_acc_epoch
# ----------------------------------------------------



# ----------------------------------------------------
#    TRAINING LOOP (iterate on epochs)
# ----------------------------------------------------
def train_model(hyper, train_loader, val_loader, network, optimizer, criterion):

    metrics = {"train_loss": [], 
               "train_acc": [],
               "val_loss": [],   
               "val_acc": []}

    for epoch in range(hyper["epochs"]):
        train_loss_epoch, train_acc_epoch = train_single_epoch(train_loader, network, optimizer, criterion)
        val_loss_epoch, val_acc_epoch = eval_single_epoch(val_loader, network, criterion)

        metrics["train_loss"].append(train_loss_epoch)
        metrics["train_acc"].append(train_acc_epoch)
        metrics["val_loss"].append(val_loss_epoch)
        metrics["val_acc"].append(val_acc_epoch)

        print(f"Epoch: {epoch+1}/{hyper['epochs']}"
            f" | loss (train/val) = {train_loss_epoch:.4f}/{val_loss_epoch:.4f}"
            f" | acc (train/val) = {train_acc_epoch:.2f}/{val_acc_epoch:.2f}")

    return metrics






# //////////////////////////////////////////////////////////////////////////////
#                     SEGMENTATION TRAINING FUNCTIONS
# //////////////////////////////////////////////////////////////////////////////

def compute_iou(labels, predictions):

    # IDENTIFY VALID LABELS (-1 is not valid)
    id_valid = labels != -1  
    labels = labels[id_valid]  
    predictions = predictions[id_valid]  

    iou = []
    for i in torch.unique(labels):
        intersection = ( (labels == i) & (predictions==i) ).sum().item()    # TOTS DOS SON i
        union = ((labels==i) | (predictions==i)).sum().item()               # O BE UN O BE L'ALTRE SON i
        iou.append(intersection / union)

    meaniou = sum(iou) / len(iou)

    return meaniou

# ----------------------------------------------------
#           UNPACK SEGMENTATION BATCH 
# ----------------------------------------------------
def unpack_segmentation_batch(batch):
    """
    Supports both:
    - PyG ModelNet batches (Data object with .pos and .y)
    - ShapeNet batches (tuple: points, object_class, seg_labels, num_seg_classes)
    Returns:
      x: [B, N, 3]
      y: [B, N]
    """
    # ---------- ShapeNet ----------
    points, object_class, seg_labels, global_labels = batch

    # Even though DataLoader will turn into torch tensors, 
    # I do this to guarantee I can use this function if I bypass DataLoader and use dataset directly
    points_BNC = torch.as_tensor(points, dtype=torch.float32)            
    labels_BN = torch.as_tensor(seg_labels, dtype=torch.long)
    
    points_BNC = points_BNC.transpose(-2,-1)

    return points_BNC, labels_BN


# ----------------------------------------------------
#    TRAINING EPOCH FUNCTION (SEGMENTATION)
# ----------------------------------------------------
def train_single_epoch_segmentation(train_loader, network, optimizer, criterion):
    
    device = next(network.parameters()).device  # guarantee that we are using the same device than the model
    network.train()                             # Activate the train=True flag inside the model

    loss_history = []
    miou_history = []
    nCorrect = 0
    nTotal = 0
    for batch in train_loader:
        
        # Pointnet needs: [batch, nPoints, coordinates] 
        points_BNC, labels_BN = unpack_segmentation_batch(batch)
        points_BNC = points_BNC.to(device)
        labels_BN = labels_BN.to(device)      

        # forward batch and loss
        optimizer.zero_grad()                  # Set network gradients to 0
        log_probs, _, _, feature_tnet, _ = network(points_BNC)    # Forward batch through the network  
        reg_loss = compute_regularizationLoss(feature_tnet)
        # NLLLoss expects class dimension at dim=1 → use [B, C, N] layout
        loss = criterion(log_probs, labels_BN) + 0.001 * reg_loss       # Compute loss: NLLLoss   
        loss_history.append(loss.item())         
        
        loss.backward()                                                 # Compute backpropagation
        optimizer.step()    
        
        # COMPUTE METRICS
        # Identify valid labels (-1 is not valid)
        id_valid = labels_BN != -1  
        num_valid = id_valid.sum().item()
        assert num_valid > 0, "All points in this batch are unlabeled (-1)!"  
        # Accuracy
        predictions = log_probs.argmax(dim=1)
        batch_correct = (predictions[id_valid] == labels_BN[id_valid]).sum().item()          # .item() brings one single scalar to CPU
        nCorrect = nCorrect + batch_correct
        nTotal = nTotal + num_valid         # segmentation: 1 prediction per point and N points
        # IoU
        miou = compute_iou(labels_BN, predictions)
        miou_history.append(miou)

    # Average across all batches    
    train_loss_epoch = np.mean(loss_history) 
    train_acc_epoch = nCorrect / nTotal
    train_miou_epoch = np.mean(miou_history)
    
    return train_loss_epoch, train_acc_epoch, train_miou_epoch
# ----------------------------------------------------


# ----------------------------------------------------
#    TESTING EPOCH FUNCTION (SEGMENTATION)
# ----------------------------------------------------
def eval_single_epoch_segmentation(data_loader, network, criterion):

    device = next(network.parameters()).device  # guarantee that we are using the same device than the model

    with torch.no_grad():
        network.eval()                      # Dectivate the train=True flag inside the model

        loss_history = []
        miou_history = []
        nCorrect = 0
        nTotal = 0
        for batch in data_loader:
            # Pointnet needs: [batch, nPoints, coordinates] 
            points_BNC, labels_BN = unpack_segmentation_batch(batch) 
            points_BNC = points_BNC.to(device)
            labels_BN = labels_BN.to(device)    

            # forward batch and loss
            log_probs, _, _, _, _ = network(points_BNC)         # Forward batch through the network
            # NLLLoss expects class dimension at dim=1 → use [B, C, N] layout
            loss = criterion(log_probs, labels_BN)                  # Compute loss
            loss_history.append(loss.item())         
            
            # COMPUTE METRICS
            # Identify valid labels (-1 is not valid)
            id_valid = labels_BN != -1  
            num_valid = id_valid.sum().item()
            assert num_valid > 0, "All points in this batch are unlabeled (-1)!"  
            # Accuracy
            predictions = log_probs.argmax(dim=1)
            batch_correct = (predictions[id_valid] == labels_BN[id_valid]).sum().item()          # .item() brings one single scalar to CPU
            nCorrect = nCorrect + batch_correct
            nTotal = nTotal + num_valid         # segmentation: 1 prediction per point and N points
            # IoU
            miou = compute_iou(labels_BN, predictions)
            miou_history.append(miou)

        # Average across all batches 
        eval_loss_epoch = np.mean(loss_history)       
        eval_acc_epoch = nCorrect / nTotal
        eval_miou_epoch = np.mean(miou_history)
    
    return eval_loss_epoch, eval_acc_epoch, eval_miou_epoch
# ----------------------------------------------------



# ----------------------------------------------------
#    TRAINING LOOP (iterate on epochs)
# ----------------------------------------------------
def train_model_segmentation(hyper, train_loader, val_loader, network, optimizer, criterion):

    metrics = {"train_loss": [],
               "train_acc": [], 
               "train_miou": [],
               "val_loss": [],   
               "val_acc": [],   
               "val_miou": []}

    for epoch in range(hyper["epochs"]):
        train_loss_epoch, train_acc_epoch, train_miou_epoch = train_single_epoch_segmentation(train_loader, network, optimizer, criterion)
        val_loss_epoch, val_acc_epoch, val_miou_epoch = eval_single_epoch_segmentation(val_loader, network, criterion)
        
        metrics["train_loss"].append(train_loss_epoch)
        metrics["train_acc"].append(train_acc_epoch)
        metrics["train_miou"].append(train_miou_epoch)
        metrics["val_loss"].append(val_loss_epoch)
        metrics["val_acc"].append(val_acc_epoch)
        metrics["val_miou"].append(val_miou_epoch)

        print(f"Epoch: {epoch+1}/{hyper['epochs']}"
            f" | loss (train/val) = {train_loss_epoch:.4f}/{val_loss_epoch:.4f}"
            f" | acc (train/val) = {train_acc_epoch:.2f}/{val_acc_epoch:.2f}"
            f" | miou (train/val) = {train_miou_epoch:.2f}/{val_miou_epoch:.2f}")

    return metrics



# //////////////////////////////////////////////////////////////////////////////
#                     GENERAL TRAINING FUNCTIONS
# //////////////////////////////////////////////////////////////////////////////

def compute_regularizationLoss(feature_tnet):
    # REGULARIZATION: force Tnet matrix to be orthogonal (TT^t = I)
    # i.e. allow transforming the sapce but without distorting it
    # The loss adds this term to be minimized: ||I-TT^t||
    # It is a training constrain --> no need to be included in validation
    TT = torch.bmm(feature_tnet, feature_tnet.transpose(2, 1))
    I = torch.eye(TT.shape[-1], device=TT.device).unsqueeze(0).expand(TT.shape[0], -1, -1) # [64,64]->[1,64,64]->[batch,64,64]
    reg_loss = torch.norm(I - TT) / TT.shape[0]                 # make reg_loss batch invariant (dividing by batch_size)

    return reg_loss
