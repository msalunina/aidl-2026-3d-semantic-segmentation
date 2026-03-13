import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from utils.config_parser import ConfigParser
from torch.utils.data import DataLoader
from utils.shapenet_dataset import shapenetDataset
from tqdm import tqdm
import json

from utils.focal_loss import FocalLoss

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from models.pointnet import PointNetSegmentation



classes_names = [ 'Airplane', 'Bag', 'Cap','Car', 
                 'Chair', 'Earphone','Guitar', 'Knife',
                 'Lamp', 'Laptop','Motorbike','Mug',
                 'Pistol','Rocket', 'Skateboard','Table']
    
classes_size = [4,2,2,4,
                4,3,3,2,
                4,2,6,2,
                3,3,3,3]

classes_folder = ['02691156','02773838','02954340','02958343',
                    '03001627','03261776','03467517','03624134',
                    '03636649','03642806','03790512','03797390',
                    '03948459','04099429','04225987','04379243']

def set_device():
    if torch.cuda.is_available(): 
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): 
        device = torch.device("mps")
    else: device = torch.device("cpu")
    1
    print(f"\nUsing device: {device}")
    return device


def compute_regularizationLoss(feature_tnet):
    # REGULARIZATION: force Tnet matrix to be orthogonal (TT^t = I)
    # i.e. allow transforming the sapce but without distorting it
    # The loss adds this term to be minimized: ||I-TT^t||
    # It is a training constrain --> no need to be included in validation

    identity = torch.eye(feature_tnet.shape[-1])
    if torch.cuda.is_available():
        identity = identity.cuda()
    regularization_loss = torch.norm(
        identity - torch.bmm(feature_tnet, feature_tnet.transpose(2, 1)))
        # Loss
    reg_loss = regularization_loss

    """
    TT = torch.bmm(feature_tnet, feature_tnet.transpose(2, 1))
    I = torch.eye(TT.shape[-1], device=TT.device).unsqueeze(0).expand(TT.shape[0], -1, -1) # [64,64]->[1,64,64]->[B,64,64]
    reg_loss = torch.norm(I - TT) / TT.shape[0]                 # make reg_loss batch invariant (dividing by batch_size)
    """
    return reg_loss


# //////////////////////////////////////////////////////////////////////////////
#                     SEGMENTATION TRAINING FUNCTIONS
# //////////////////////////////////////////////////////////////////////////////

def compute_batch_intersection_and_union(labels, predictions, num_classes, iou_offset, object_class): 

    iou_batch = torch.zeros(num_classes, device=predictions.device, dtype=torch.float32)

    for cnt, i in enumerate(object_class): #objects ids in the batch
        #segmentation classes for each object
        seg_classes = shapenetDataset.seg_classes[classes_names[i]]
        #IoU
        inter_batch = 0
        union_batch = 0
        iou_object = torch.zeros(len(seg_classes), device=predictions.device, dtype=torch.float32)
        for j, c in enumerate(seg_classes):
            inter_batch = ((predictions[cnt] == c) & (labels[cnt] == c)).sum()     # BOTH ARE c
            union_batch = ((predictions[cnt] == c) | (labels[cnt] == c)).sum()     # EITHER ONE OR THE OTHER ARE c
            
            if(union_batch > 0):
                iou_object[j] = (inter_batch/union_batch)/len(seg_classes)
        
        iou_batch[i] += iou_object.sum() + iou_offset[cnt]


    return iou_batch


# ----------------------------------------------------
#    TRAINING EPOCH FUNCTION (SEGMENTATION)
# ----------------------------------------------------
def train_single_epoch_segmentation(config, train_loader, network:PointNetSegmentation, optimizer, criterion, objects_count):
    
    device = next(network.parameters()).device  # guarantee that we are using the same device than the model
    network.train()                             # Activate the train=True flag inside the model

    loss_history = []
    nCorrect = 0
    nTotal = 0
    train_iou = torch.zeros( len(classes_names), device=device, dtype=torch.float32)
    objects_count = objects_count.to(device)
    
    for batch in tqdm(train_loader, desc="train epoch", position=1, leave=False):

        # Pointnet needs: [B, N, C]
        points_BNC, labels, iou_offset, object_class = batch                           # Points: [B, N, C] / labels: [B, N] / image [B, H, W]       
        points_BNC = points_BNC.to(device)
        labels = labels.to(device) 

        # Set network gradients to 0
        optimizer.zero_grad()  

        network.setOneHotVectorBatch(object_class)

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
        
        loss.backward()                                           
        optimizer.step()    

        # ----------- COMPUTE METRICS -------------
        # Compute predictions
        predictions = log_probs_BCN.argmax(dim=1)
        # Identify valid labels (-1 is not valid)
  
        # Accuracy
        batch_correct = (predictions == labels).sum().item()    # num correct (valid) per batch 
        nCorrect += batch_correct                                 # num correct (valid) per epoch 
        nTotal += (labels.shape[0] * labels.shape[1])                               # num total (valid) per epoch 
        # Update intersection and union
        iou_batch = compute_batch_intersection_and_union(labels, predictions, len(classes_names), iou_offset, object_class)
        train_iou += iou_batch
        # ------------------------------------------

    assert nTotal > 0, "No valid points in epoch (all labels are -1)."
   
    # Average across all batches    
    train_loss_epoch = np.mean(loss_history) 
    train_acc_epoch = nCorrect / nTotal
    # Compute IoU per class and mean
    
    train_iou /= objects_count

    return train_loss_epoch, train_acc_epoch, train_iou
# ----------------------------------------------------



# ----------------------------------------------------
#    TESTING EPOCH FUNCTION (SEGMENTATION)
# ----------------------------------------------------
def eval_single_epoch_segmentation(config, data_loader, network:PointNetSegmentation, criterion, objects_count):

    device = next(network.parameters()).device  # guarantee that we are using the same device than the model
    objects_count = objects_count.to(device)

    with torch.no_grad():
        network.eval()                      # Dectivate the train=True flag inside the model

        loss_history = []
        nCorrect = 0
        nTotal = 0
        eval_iou = torch.zeros(len(classes_names), device=device, dtype=torch.float32)

        for batch in tqdm(data_loader, desc="val epoch", position=1, leave=False):
            
            # Pointnet needs: [B, N, C]   
            points_BNC, labels, iou_offset, object_class = batch  # Points: [B, N, C] / labels: [B, N] / image [B, H, W]       
            points_BNC = points_BNC.to(device)
            labels = labels.to(device) 

            network.setOneHotVectorBatch(object_class)

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
           
            # Accuracy
            batch_correct = (predictions == labels).sum().item()    # num correct (valid) per batch 
            nCorrect += batch_correct                                 # num correct (valid) per epoch 
            nTotal += (labels.shape[0] * labels.shape[1])                                 # num total (valid) per epoch 
            # Update intersection and union
            iou_batch = compute_batch_intersection_and_union(labels, predictions, len(classes_names), iou_offset, object_class)
            eval_iou += iou_batch
          
            # ------------------------------------------
        
        assert nTotal > 0, "No valid points in epoch (all labels are -1)."

        # Average across all batches    
        eval_loss_epoch = np.mean(loss_history) 
        eval_acc_epoch = nCorrect / nTotal
        # Compute IoU per class and mean
        
        eval_iou /= objects_count

    return eval_loss_epoch, eval_acc_epoch, eval_iou
# ----------------------------------------------------

# ----------------------------------------------------
#    TESTING FUNCTION test split
# ----------------------------------------------------
def test_single_epoch_segmentation(data_loader, network:PointNetSegmentation, objects_count):

    device = next(network.parameters()).device  # guarantee that we are using the same device than the model
    objects_count = objects_count.to(device)

    with torch.no_grad():
        network.eval()                      # Dectivate the train=True flag inside the model

        nCorrect = 0
        nTotal = 0
        test_iou = torch.zeros(len(classes_names), device=device, dtype=torch.float32)

        for batch in tqdm(data_loader, desc="test epoch", position=1, leave=False):
            
            # Pointnet needs: [B, N, C]   
            points_BNC, labels, iou_offset, object_class = batch  # Points: [B, N, C] / labels: [B, N] / image [B, H, W]       
            points_BNC = points_BNC.to(device)
            labels = labels.to(device) 
            
            network.setOneHotVectorBatch(object_class)

            output = network(points_BNC)   

            # Handle output depending on what model returns
            if isinstance(output, tuple):
                feature_tnet, log_probs_BCN = output
            else:
                log_probs_BCN = output
   
            
            # ----------- COMPUTE METRICS -------------
            # Compute predictions
            predictions = log_probs_BCN.argmax(dim=1)
            # Identify valid labels (-1 is not valid)
           
            # Accuracy
            batch_correct = (predictions == labels).sum().item()    # num correct (valid) per batch 
            nCorrect += batch_correct                                 # num correct (valid) per epoch 
            nTotal += (labels.shape[0] * labels.shape[1])                                 # num total (valid) per epoch 
            # Update intersection and union
            iou_batch = compute_batch_intersection_and_union(labels, predictions, len(classes_names), iou_offset, object_class)
            test_iou += iou_batch
          
            # ------------------------------------------
        
        assert nTotal > 0, "No valid points in epoch (all labels are -1)."

        # Average across all batches    
        test_acc_epoch = nCorrect / nTotal
        # Compute IoU per class and mean
        
        test_iou /= objects_count

    return test_acc_epoch, test_iou

# ----------------------------------------------------
#    TEST IOU FUNCTION (SEGMENTATION)
# ----------------------------------------------------
def test_iou_values(config, model:PointNetSegmentation,  train_loader, objects_count, device):
    
    nTotal = 0
    objects_count.cpu()
    train_iou = torch.zeros( len(classes_names), device=device, dtype=torch.float32)
    nCorrect = 0
    for batch in tqdm(train_loader, desc="train epoch", position=1, leave=False):

        # Pointnet needs: [B, N, C]
        points, labels, iou_offset, object_class = batch                           # Points: [B, N, C] / labels: [B, N] / image [B, H, W]       
        predictions = labels     
        points = points.to(device)
        predictions = predictions.to(device)
        labels = labels.to(device)
        iou_offset = iou_offset.to(device)
        object_class = object_class.to(device)  

        model.setOneHotVectorBatch(labels)

        feat_tf, pred = model(points)
        # Accuracy
        batch_correct = (predictions == labels)
        batch_correct = batch_correct.sum().item() 
        nCorrect = nCorrect + batch_correct                                 # num correct (valid) per epoch 
        nTotal += (labels.shape[0] * labels.shape[1])               # num total (valid) per epoch 
        # Update intersection and union
        iou_batch = compute_batch_intersection_and_union(labels, predictions, len(classes_names), iou_offset, object_class)
        train_iou += iou_batch
        # ------------------------------------------   
    
    train_iou /= objects_count
    test_acc_epoch = nCorrect / nTotal
    return train_iou, test_acc_epoch
# ----------------------------------------------------

def getLossWeightsForDataset(path):

    #get the weigths for the training split
    points_per_class = np.zeros(50)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for path in data:
        name = path.replace("shape_data","./data/ShapeNet/raw") + ".txt"
        
        with open(name, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                label = int(float(line.split(" ")[-1]))
                points_per_class[label]+=1
        
    #compute the weights compute_focal_weights(points_per_class, points_per_class.sum(), 50)
    weights_per_class = points_per_class.sum() / (50 * points_per_class)
    return weights_per_class
        
def countObjectsSplit(path):
    objects_per_class = np.zeros(len(classes_names))

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for path in data:
        name = path.replace("shape_data","./data/ShapeNet/raw") + ".txt"

        for i, class_id in enumerate(classes_folder):
            if class_id in name:
                objects_per_class[i] +=1
                break
    
    return objects_per_class


def train_shapenet(loader_train, loader_eval, loader_test, config):

    # Set device
    device = set_device()

    base_path = Path(os.getcwd())
    if "src" in base_path.parts:
        base_path = base_path[:-1]

    train_objects_count = countObjectsSplit(os.path.join(base_path, "data/ShapeNet/raw", "train_test_split","shuffled_train_file_list.json"))
    val_objects_count = countObjectsSplit(os.path.join(base_path, "data/ShapeNet/raw", "train_test_split","shuffled_val_file_list.json"))
    test_objects_count = countObjectsSplit(os.path.join(base_path, "data/ShapeNet/raw", "train_test_split","shuffled_test_file_list.json"))

    """
    results_str =    "        |" 
    total_str =     f" TOTAL  |"
    results_train = f" TRAIN  |"
    results_eval  = f"  EVAL  |"
    results_test  = f"  TEST  |"

    for n in range(len(classes_names)):
        total = train_objects_count[n] + val_objects_count[n] + test_objects_count[n]
        total_str +=     f"  { total }                 |"
        results_str +=   f"  {classes_names[n][:3]}    |"
        results_train += f"  {train_objects_count[n]}  ({(train_objects_count[n]/total)*100.0:.2f} % )  |"
        results_eval +=  f"  {val_objects_count[n]}  ({(val_objects_count[n]/total) *100.0:.2f} % )    |" 
        results_test +=  f"  {test_objects_count[n]}  ({(test_objects_count[n]/total) *100.0:.2f} % ) |"   

    print(len(results_str)*'*')
    print("OBJECTS COUNT BY SPLIT\n")
    print(results_str)
    print(len(results_str)*'-')
    print(results_train)
    print(results_eval)
    print(results_test)
    print("\n")
    print(len(results_str)*'*')
    """

    train_objects_count = torch.from_numpy(train_objects_count).to(device)
    val_objects_count = torch.from_numpy(val_objects_count).to(device)
    test_objects_count = torch.from_numpy(test_objects_count).to(device)

    #loss_weights = getLossWeightsForDataset(os.path.join(base_path,"train_test_split","shuffled_train_file_list.json"))

    """COUNTS FOR POINTS in test split"""
    class_counts = np.array([2.275419e+06, 1.664032e+06, 6.421560e+05, 4.605680e+05,
                            9.715000e+03, 1.381380e+05, 7.548700e+04, 2.672600e+04,
                            9.784400e+04, 1.315100e+05, 2.887660e+05, 1.302657e+06,
                            2.430900e+06, 2.989559e+06, 1.510814e+06, 2.602980e+05,
                            7.497200e+04, 3.148200e+04, 1.568300e+04, 1.096760e+05,
                            2.460910e+05, 9.421160e+05, 2.987310e+05, 2.898890e+05,
                            3.571130e+05, 1.535899e+06, 3.095600e+04, 5.066720e+05,
                            4.790800e+05, 4.152520e+05, 1.596000e+04, 1.361300e+04,
                            8.020200e+04, 5.217000e+03, 2.217000e+03, 2.238510e+05,
                            2.232500e+04, 3.436340e+05, 3.599710e+05, 1.645930e+05,
                            3.052800e+04, 7.660800e+04, 1.951900e+04, 1.256100e+04,
                            2.949200e+04, 2.187040e+05, 1.882900e+04, 7.729973e+06,
                            2.395978e+06, 3.172600e+05])
    
    offset = 0
    loss_weights = np.zeros(50)
    imbalances = []
    for _, part_labels in shapenetDataset.seg_classes.items():

        end_pos =  offset + len(part_labels)
        imbalances.append(class_counts[offset:end_pos].max() / class_counts[offset:end_pos].min())
        loss_weights[offset:end_pos] = class_counts[offset:end_pos].sum() / (len(part_labels) * class_counts[offset:end_pos])   
        offset += len(part_labels)

    imbalances = np.array(imbalances, dtype=np.float32)
    
    print(f"Mean imbalance {imbalances.sum()/len(imbalances)}\nImbalance for class object {imbalances}")
    
    """Direct inverse
    loss_weights = np.array([2.78288265e-01, 3.93399671e-01, 9.92885292e-01, 1.37892433e+00,
    5.76229184e+01, 4.55120575e+00, 8.27582472e+00, 2.30545934e+01,
    6.71087918e+00, 4.98647959e+00, 2.21998937e+00, 4.96067403e-01,
    2.54424988e-01, 2.11409835e-01, 4.04915845e-01, 2.72325381e+00,
    8.22109615e+00, 1.92900324e+01, 4.55080628e+01, 5.49545886e+00,
    2.41758346e+00, 6.63925415e-01, 2.09398996e+00, 2.07653581e+00,
    1.82575604e+00, 4.03944003e-01, 2.05516861e+01, 1.27633717e+00,
    1.33946507e+00, 1.54831530e+00, 2.96069923e+01, 4.67660721e+01,
    7.08281898e+00, 1.18995566e+02, 2.04881488e+02, 2.39619002e+00,
    2.87035581e+01, 1.80757393e+00, 1.82148116e+00, 4.25819807e+00,
    2.24646402e+01, 8.03024927e+00, 3.05458382e+01, 5.16889608e+01,
    2.01781131e+01, 2.80746287e+00, 3.28133785e+01, 8.41115039e-02,
    2.67126574e-01, 1.60079651e+00], dtype=np.float32)
    """
    
    """
    loss_weights = np.array([0.17754055, 0.21108912, 0.33534504, 0.39519221, 2.55109139, 0.71790496,
                            0.96798568,  1.61503408, 0.8717059,  0.75144295, 0.50142325, 0.23703806,
                            0.169758,    0.15474386, 0.21415643, 0.55535108, 0.96478101, 1.4774414,
                            2.26779484,  0.78885185, 0.5232601,  0.27422406, 0.48698736, 0.48495372,
                            0.45473112,  0.21389928, 1.52494377, 0.38020864, 0.38949714, 0.4187608,
                            1.82990715,  2.2988541,  0.89552829, 3.66041533, 4.79282232, 0.52094004,
                            1.8018124,   0.4524614,  0.4541985,  0.69441617, 1.59425978, 0.95352145,
                            1.85865059,  2.4165264,  1.51103468, 0.56387087, 1.92629423, 0.09760676,
                            0.17394373,  0.42579819], dtype=np.float32)
    """

   
    model = PointNetSegmentation(num_classes=50, # 50 shape classes 
                                input_channels=3, 
                                dropout=0.3,
                                skip_conn=True,
                                add_ohv=True).to(device) #0.3 dropout


    loss_weights = torch.tensor(loss_weights, dtype=torch.float32).to(device)
    criterion = nn.NLLLoss()#weight=loss_weights) #FocalLoss(alpha=loss_weights, gamma=0.5, ignore_index=config.ignore_label)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.num_epochs,
            eta_min=0.00001
    )

    
    #TEST 
    #iou, acc = test_iou_values(config, model, loader_eval, val_objects_count, device)
    
    test_name = f"{config.test_name}_{config.num_epochs}"

    writer = SummaryWriter(log_dir=f"./logs/{test_name}_ep")
    

    for epoch in range(config.num_epochs):
              
        train_loss_epoch, train_acc_epoch, train_iou_class_epoch = train_single_epoch_segmentation(config, loader_train, model, optimizer, criterion, train_objects_count)

        val_loss_epoch, val_acc_epoch, val_iou_class_epoch = eval_single_epoch_segmentation(config, loader_test, model, criterion, test_objects_count)
        
        scheduler.step()
        
        tqdm.write(f" Epoch: {epoch+1}/{config.num_epochs}"
            f" | loss (train/val) = {train_loss_epoch:.3f}/{val_loss_epoch:.3f}"
            f" | acc (train/val) = {train_acc_epoch:.3f}/{val_acc_epoch:.3f}"
            f" | mIoU (train/val) = {train_iou_class_epoch.sum()/len(train_iou_class_epoch):.3f}/ {val_iou_class_epoch.sum() / len(val_iou_class_epoch):3f}")
    
        writer.add_scalar(tag=f"{test_name}_Train/Loss", scalar_value=train_loss_epoch, global_step=epoch)
        writer.add_scalar(tag=f"{test_name}_Train/Accuracy", scalar_value=train_acc_epoch, global_step=epoch)
        writer.add_scalar(tag=f"{test_name}_Train/mIoU", scalar_value=(train_iou_class_epoch.sum()/len(train_iou_class_epoch) ), global_step=epoch)
        writer.add_scalar(tag=f"{test_name}_Validation/Loss", scalar_value=val_loss_epoch, global_step=epoch)
        writer.add_scalar(tag=f"{test_name}_Validation/Accuracy", scalar_value=val_acc_epoch, global_step=epoch)
        writer.add_scalar(tag=f"{test_name}_Validation/mIoU", scalar_value=(val_iou_class_epoch.sum() / len(val_iou_class_epoch)), global_step=epoch)

        # Add per-class IoU metrics

        for i in range(len(classes_names)):
            writer.add_scalar(tag=f"{test_name}_IoU_Class/{classes_names[i]}/Train", scalar_value=train_iou_class_epoch[i], global_step=epoch)
            writer.add_scalar(tag=f"{test_name}_IoU_Class/{classes_names[i]}/Validation", scalar_value=val_iou_class_epoch[i], global_step=epoch)

    
    #do a test run
    test_acc_epoch, test_iou_class_epoch = test_single_epoch_segmentation(loader_eval, model, val_objects_count)

    results_str =    "        |  MEAN  |"
    results_train = f" TRAIN  |  {(train_iou_class_epoch.sum()/len(train_iou_class_epoch)):.2f}  |"
    results_eval  = f"  EVAL  |  {(val_iou_class_epoch.sum()/len(val_iou_class_epoch)):.2f}  |"
    results_test  = f"  TEST  |  {(test_iou_class_epoch.sum()/len(test_iou_class_epoch)):.2f}  |"

    for n in range(len(classes_names)):
        results_str += f"   {classes_names[n][:3]}   |"
        results_train += f"  {train_iou_class_epoch[n]:.3f}  |"
        results_eval += f"  {val_iou_class_epoch[n]:.3f}  |" 
        results_test += f"  {test_iou_class_epoch[n]:.3f}  |"   
    
    print(len(results_str)*'*')
    print("FINAL RESULTS\n")
    print(results_str)
    print(len(results_str)*'-')
    print(results_train)
    print(results_eval)
    print(results_test)
    print("\n")
    print(len(results_str)*'*')

    checkpoint = {"model_state_dict": model.cpu().state_dict()}
    path_to_save = os.path.join("./snapshots",f"shapenet_{test_name}.pt")
    torch.save(checkpoint, path_to_save)

def test_shapenet(loader_train, loader_eval, loader_test, config):
    
    base_path = Path(os.getcwd())
    if "src" in base_path.parts:
        base_path = base_path[:-1]

    # Set device
    device = set_device()

    train_objects_count = countObjectsSplit(os.path.join(base_path, "data/ShapeNet/raw", "train_test_split","shuffled_train_file_list.json"))
    val_objects_count = countObjectsSplit(os.path.join(base_path, "data/ShapeNet/raw","train_test_split","shuffled_val_file_list.json"))
    test_objects_count = countObjectsSplit(os.path.join(base_path, "data/ShapeNet/raw", "train_test_split","shuffled_test_file_list.json"))

    train_objects_count = torch.from_numpy(train_objects_count).to(device)
    val_objects_count = torch.from_numpy(val_objects_count).to(device)
    test_objects_count = torch.from_numpy(test_objects_count).to(device)

    print("\n" + "="*60)
    print("CREATING DATASETS AND DATALOADERS")
    print("="*60)
    
    model_trained = PointNetSegmentation(num_classes=50, # 50 shape classes 
                                input_channels=3, 
                                dropout=0.3,
                                add_ohv=True,
                                skip_conn=True).to(device) #0.3 dropout

     
    checkpoint_path= os.path.join(base_path, "snapshots", "shapenet_PointNet_shapenet_xyz_all_50.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)    # it loads more things that weights

    # UPDATE ARCHITECTURE
    model_trained.load_state_dict(checkpoint["model_state_dict"])
    model_trained.to(device)
    model_trained.eval()      # changes behaviour of some layers (e.g. dropout off, batchnorm), does not strop gradient
 
    train_acc_epoch, train_iou_class_epoch = test_single_epoch_segmentation(loader_train, model_trained, train_objects_count)

    val_acc_epoch, val_iou_class_epoch = test_single_epoch_segmentation(loader_test, model_trained, test_objects_count)

    test_acc_epoch, test_iou_class_epoch = test_single_epoch_segmentation(loader_eval, model_trained, val_objects_count)

    results_str =    "        |  MEAN  |"
    results_train = f" TRAIN  |  {(train_iou_class_epoch.sum()/len(train_iou_class_epoch)):.2f}  |"
    results_eval  = f"  EVAL  |  {(val_iou_class_epoch.sum()/len(val_iou_class_epoch)):.2f}  |"
    results_test  = f"  TEST  |  {(test_iou_class_epoch.sum()/len(test_iou_class_epoch)):.2f}  |"

    for n in range(len(classes_names)):
        results_str += f"   {classes_names[n][:3]}   |"
        results_train += f"  {train_iou_class_epoch[n]:.3f}  |"
        results_eval += f"  {val_iou_class_epoch[n]:.3f}  |" 
        results_test += f"  {test_iou_class_epoch[n]:.3f}  |"   
    
    print(len(results_str)*'*')
    print("FINAL RESULTS\n")
    print(results_str)
    print(len(results_str)*'-')
    print(results_train)
    print(results_eval)
    print(results_test)
    print("\n")
    print(len(results_str)*'*')


if __name__ == '__main__':

    base_path = "./data/ShapeNet/raw"

    config_parser = ConfigParser(
        default_config_path="config/pointnet_shapenet.yaml",
        parser=argparse.ArgumentParser(description='3D Semantic Segmentation on DALES Dataset')
    )
    config = config_parser.load()

    pc_size = config.train_num_points

    # Create test datasets
    dataset_train = shapenetDataset(base_path, "train", pc_size, 0.7)
    dataset_eval = shapenetDataset(base_path, "eval",  pc_size, 0.7)
    dataset_test = shapenetDataset(base_path, "test",  pc_size, 0.0)

    loader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    loader_eval = DataLoader(dataset_eval, batch_size=config.batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=True)


    train_shapenet(loader_train, loader_eval, loader_test, config)

    #test_shapenet(loader_train, loader_eval, loader_test, config)