from tqdm import tqdm
from utils.utils import loss_function
import torch
import matplotlib.pyplot as plt


def train_single_epoch(model, optimizer, data_loader, device, alpha=0.001):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Reshape from (batch_size * num_points, 3) to (batch_size, num_points, 3)
        batch_size = data.num_graphs
        num_points = data.pos.shape[0] // batch_size
        pos_reshaped = data.pos.view(batch_size, num_points, 3)

        feature_transform, predictions = model(pos_reshaped)
        loss = loss_function(predictions, data.y,
                             feature_transform, alpha=alpha)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

        # Calculate accuracy
        pred_labels = predictions.argmax(dim=1)
        correct += (pred_labels == data.y).sum().item()
        total += data.y.size(0)

    train_loss = total_loss / len(data_loader.dataset)
    train_acc = correct / total
    return train_loss, train_acc


def eval_single_epoch(model, data_loader, device, alpha=0.001):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)

            # Reshape from (batch_size * num_points, 3) to (batch_size, num_points, 3)
            batch_size = data.num_graphs
            num_points = data.pos.shape[0] // batch_size
            pos_reshaped = data.pos.view(batch_size, num_points, 3)

            feature_transform, predictions = model(pos_reshaped)
            loss = loss_function(predictions, data.y,
                                 feature_transform, alpha=alpha)
            total_loss += loss.item() * data.num_graphs

            # Calculate accuracy
            pred_labels = predictions.argmax(dim=1)
            correct += (pred_labels == data.y).sum().item()
            total += data.y.size(0)

    val_loss = total_loss / len(data_loader.dataset)
    val_acc = correct / total
    return val_loss, val_acc


def train_model(model, optimizer, train_loader, val_loader, config, model_save_path=None):
    print("Training model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    best_val_loss = float('inf')

    for epoch in tqdm(range(1, config["num_epochs"] + 1)):
        epoch_train_loss, epoch_train_acc = train_single_epoch(
            model, optimizer, train_loader, device, alpha=config.get("alpha", 0.001))
        epoch_val_loss, epoch_val_acc = eval_single_epoch(
            model, val_loader, device, alpha=config.get("alpha", 0.001))
        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss)
        train_acc_history.append(epoch_train_acc)
        val_acc_history.append(epoch_val_acc)

        print(f'\nEpoch: {epoch:03d}, Train Loss: {epoch_train_loss:.4f}, \
              Val Loss: {epoch_val_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f}')
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)

    print("Training complete.")
    return train_loss_history, val_loss_history, train_acc_history, val_acc_history


def plot_losses(train_loss, test_loss, save_to_file=None):
    fig = plt.figure()
    epochs = len(train_loss)
    plt.plot(range(epochs), train_loss, 'b', label='Training loss')
    plt.plot(range(epochs), test_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    if save_to_file:
        fig.savefig(save_to_file, dpi=200)


def plot_accuracies(train_acc, test_acc, save_to_file=None):
    fig = plt.figure()
    epochs = len(train_acc)
    plt.plot(range(epochs), train_acc, 'b', label='Training accuracy')
    plt.plot(range(epochs), test_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    if save_to_file:
        fig.savefig(save_to_file)


def plot_ious(train_iou, test_iou, save_to_file=None):
    fig = plt.figure()
    epochs = len(train_iou)
    plt.plot(range(epochs), train_iou, 'b', label='Training IoU')
    plt.plot(range(epochs), test_iou, 'r', label='Validation IoU')
    plt.title('Training and validation IoU')
    plt.legend()
    if save_to_file:
        fig.savefig(save_to_file)


def calculate_iou(predictions, targets, num_classes):
    """
    Calculate mean IoU (Intersection over Union) for segmentation.
    
    Args:
        predictions: Predicted labels (batch_size * num_points,)
        targets: Ground truth labels (batch_size * num_points,)
        num_classes: Number of segmentation classes
    
    Returns:
        mean_iou: Mean IoU across all classes
    """
    ious = []
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    for cls in range(num_classes):
        pred_mask = (predictions == cls)
        target_mask = (targets == cls)
        
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        
        if union == 0:
            # If this class doesn't appear in ground truth, skip it
            continue
        
        iou = intersection / union
        ious.append(iou)
    
    return sum(ious) / len(ious) if ious else 0.0


def train_segmentation_single_epoch(model, optimizer, data_loader, device, num_classes, alpha=0.001):
    """
    Train the segmentation model for one epoch.
    
    Args:
        model: PointNet segmentation model
        optimizer: Optimizer
        data_loader: Training data loader
        device: Device (cuda or cpu)
        num_classes: Number of segmentation classes
        alpha: Regularization weight for feature transform
    
    Returns:
        train_loss: Average training loss
        train_acc: Average training accuracy
        train_iou: Average training IoU
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    for pointcloud, pc_class, label, seg_class in tqdm(data_loader, desc="Training", leave=False):
        pointcloud = pointcloud.to(device)
        seg_class = seg_class.to(device)
        
        # Transpose from (batch, 3, num_points) to (batch, num_points, 3)
        pointcloud = pointcloud.transpose(2, 1)
        
        optimizer.zero_grad()
        
        # Forward pass
        feature_transform, predictions = model(pointcloud)
        
        # Reshape predictions and labels
        # predictions: (batch_size, num_classes, num_points)
        # label: (batch_size, num_points)
        predictions = predictions.transpose(2, 1).contiguous()
        predictions = predictions.view(-1, num_classes)
        seg_class = seg_class.view(-1)
        
        # Calculate loss
        loss = loss_function(predictions, seg_class, feature_transform, alpha=alpha)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * pointcloud.size(0)
        
        # Calculate accuracy
        pred_labels = predictions.argmax(dim=1)
        correct += (pred_labels == seg_class).sum().item()
        total += seg_class.size(0)
        
        # Store for IoU calculation - move to CPU to save GPU memory
        all_predictions.append(pred_labels.cpu())
        all_targets.append(seg_class.cpu())
    
    train_loss = total_loss / len(data_loader.dataset)
    train_acc = correct / total
    
    # Calculate IoU
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    train_iou = calculate_iou(all_predictions, all_targets, num_classes)
    
    return train_loss, train_acc, train_iou


def eval_segmentation_single_epoch(model, data_loader, device, num_classes, alpha=0.001):
    """
    Evaluate the segmentation model for one epoch.
    
    Args:
        model: PointNet segmentation model
        data_loader: Validation/test data loader
        device: Device (cuda or cpu)
        num_classes: Number of segmentation classes
        alpha: Regularization weight for feature transform
    
    Returns:
        val_loss: Average validation loss
        val_acc: Average validation accuracy
        val_iou: Average validation IoU
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for pointcloud, pc_class, label, seg_class in tqdm(data_loader, desc="Validation", leave=False):
            pointcloud = pointcloud.to(device)
            seg_class = seg_class.to(device)
            
            # Transpose from (batch, 3, num_points) to (batch, num_points, 3)
            pointcloud = pointcloud.transpose(2, 1)
            
            # Forward pass
            feature_transform, predictions = model(pointcloud)
            
            # Reshape predictions and labels
            predictions = predictions.transpose(2, 1).contiguous()
            predictions = predictions.view(-1, num_classes)
            seg_class = seg_class.view(-1)
            
            # Calculate loss
            loss = loss_function(predictions, seg_class, feature_transform, alpha=alpha)
            total_loss += loss.item() * pointcloud.size(0)
            
            # Calculate accuracy
            pred_labels = predictions.argmax(dim=1)
            correct += (pred_labels == seg_class).sum().item()
            total += seg_class.size(0)
            
            # Store for IoU calculation - move to CPU to save GPU memory
            all_predictions.append(pred_labels.cpu())
            all_targets.append(seg_class.cpu())
    
    val_loss = total_loss / len(data_loader.dataset)
    val_acc = correct / total
    
    # Calculate IoU
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    val_iou = calculate_iou(all_predictions, all_targets, num_classes)
    
    return val_loss, val_acc, val_iou


def train_segmentation_model(model, optimizer, train_loader, val_loader, config, model_save_path=None):
    """
    Train the segmentation model.
    
    Args:
        model: PointNet segmentation model
        optimizer: Optimizer
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        model_save_path: Path to save the best model
    
    Returns:
        train_loss_history: List of training losses
        val_loss_history: List of validation losses
        train_acc_history: List of training accuracies
        val_acc_history: List of validation accuracies
        train_iou_history: List of training IoUs
        val_iou_history: List of validation IoUs
    """
    print("Training segmentation model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    model.to(device)
    
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    train_iou_history = []
    val_iou_history = []
    best_val_iou = 0.0
    
    for epoch in tqdm(range(1, config["num_epochs"] + 1)):
        epoch_train_loss, epoch_train_acc, epoch_train_iou = train_segmentation_single_epoch(
            model, optimizer, train_loader, device, 
            config["num_classes"], alpha=config.get("alpha", 0.001))
        epoch_val_loss, epoch_val_acc, epoch_val_iou = eval_segmentation_single_epoch(
            model, val_loader, device, 
            config["num_classes"], alpha=config.get("alpha", 0.001))
        
        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss)
        train_acc_history.append(epoch_train_acc)
        val_acc_history.append(epoch_val_acc)
        train_iou_history.append(epoch_train_iou)
        val_iou_history.append(epoch_val_iou)
        
        print(f'\nEpoch: {epoch:03d}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        print(f'Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f}')
        print(f'Train IoU: {epoch_train_iou:.4f}, Val IoU: {epoch_val_iou:.4f}')
        
        # Save best model based on validation IoU
        if epoch_val_iou > best_val_iou:
            best_val_iou = epoch_val_iou
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
                print(f'Model saved with Val IoU: {best_val_iou:.4f}')
    
    print("Training complete.")
    return train_loss_history, val_loss_history, train_acc_history, val_acc_history, train_iou_history, val_iou_history
