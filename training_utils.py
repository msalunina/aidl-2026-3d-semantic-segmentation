import torch
from torch_geometric.utils import to_dense_batch
import numpy as np


# Cuda agnostic thingy
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# ----------------------------------------------------
#    TRAINING EPOCH FUNCTION (iterate on data_loader)
# ----------------------------------------------------
def train_single_epoch(train_loader, network, optimizer, criterion):
    
    device = next(network.parameters()).device  # guarantee that we are using the same device than the model
    network.train()                             # Activate the train=True flag inside the model

    loss_history = []
    nCorrect = 0
    nTotal = 0
    for batch_id, batch in enumerate(train_loader):

        batch = batch.to(device)
        label = batch.y                                                 
        # Pointnet needs: [object, nPoints, coordinades] 
        # i.e. [32 object, 1024 points, 3 coordinates]: [batch_size, nPoints, 3]
        # Since batch.batch is [32x1024, 3] we have to split it into individual object points.
        # It could be done manually but "to_dense_batch" does that.
        # "to_dense_batch" will also add padding if not all objects have same number of points. In our case they have.
        # mask is a [batch_size, nPoints] boolean saying if an entry is actually a real point or padding
        BatchPointsCoords, _ = to_dense_batch(batch.pos, batch.batch)   

        optimizer.zero_grad()                  # Set network gradients to 0
        log_probs, _, _, feature_tnet_tensor, input_tnet_tensor = network(BatchPointsCoords)    # Forward batch through the network  
        # REGULARIZATION: force Tnet matrix to be orthogonal (TT^t = I)
        # i.e. allow transforming the sapce but without distorting it
        # The loss adds this term to be minimized: ||I-TT^t||
        # It is a training constrain --> no need to be included in validation
        TT = torch.bmm(feature_tnet_tensor, feature_tnet_tensor.transpose(2, 1))
        I = torch.eye(TT.shape[-1], device=TT.device).unsqueeze(0).expand(TT.shape[0], -1, -1) # [64,64]->[1,64,64]->[batch,64,64]
        reg_loss = torch.norm(I - TT) / TT.shape[0]                 # make reg_loss batch invariant (dividing by batch_size)
        loss = criterion(log_probs, label) + 0.001 * reg_loss       # Compute loss: NLLLoss   
        loss.backward()                                             # Compute backpropagation
        optimizer.step()    
        # Compute metrics
        loss_history.append(loss.item())         
        prediction = log_probs.argmax(dim=1)
        batch_correct = (prediction == label).sum().item()          # .item() brings one single scalar to CPU
        nCorrect = nCorrect + batch_correct
        nTotal = nTotal + len(label)

        if batch_id == 0:
            with torch.no_grad():
                svals = torch.linalg.svdvals(input_tnet_tensor)  # [B, 3]
                print("Input T-Net singular values (batch average) :", svals.mean(dim=0))

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
            batch = batch.to(device)

            label = batch.y 
            BatchPointsCoords, _ = to_dense_batch(batch.pos, batch.batch)   

            log_probs, _, _, _, _ = network(BatchPointsCoords)  # Forward batch through the network
            loss = criterion(log_probs, label)                  # Compute loss
            # Compute metrics
            loss_history.append(loss.item())         
            prediction = log_probs.argmax(dim=1)
            batch_correct = (prediction == label).sum().item()  # .item() brings one single scalar to CPU
            nCorrect = nCorrect + batch_correct
            nTotal = nTotal + len(label)
        
        # Average across all batches 
        eval_loss_epoch = np.mean(loss_history)       
        eval_acc_epoch = nCorrect / nTotal
    
    return eval_loss_epoch, eval_acc_epoch
# ----------------------------------------------------



# ----------------------------------------------------
#    TRAINING LOOP (iterate on epochs)
# ----------------------------------------------------
def train_model(config, train_loader, val_loader, network, optimizer, criterion, save_path=None):

    train_loss=[]
    train_acc=[]
    val_loss=[]
    val_acc=[]

    for epoch in range(config["epochs"]):
        train_loss_epoch, train_acc_epoch = train_single_epoch(train_loader, network, optimizer, criterion)
        val_loss_epoch, val_acc_epoch = eval_single_epoch(val_loader, network, criterion)
        
        train_loss.append(train_loss_epoch)
        train_acc.append(train_acc_epoch)
        val_loss.append(val_loss_epoch)
        val_acc.append(val_acc_epoch)

        print(f'Epoch: {epoch+1}/{config["epochs"]} | loss (train/val) = {train_loss_epoch:.4f}/{val_loss_epoch:.4f} | acc (train/val) ={train_acc_epoch:.2f}/{val_acc_epoch:.2f}')
    
    if save_path:
        torch.save({"model": network.state_dict(),
                    "epochs": config["epochs"], 
                    "nPoints": config["nPoints"]},
                    save_path)

    return train_loss, train_acc, val_loss, val_acc