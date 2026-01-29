import torch


def predict(model, data_loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    all_predictions = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            
            # Reshape from (batch_size * num_points, 3) to (batch_size, num_points, 3)
            batch_size = data.num_graphs
            num_points = data.pos.shape[0] // batch_size
            pos_reshaped = data.pos.view(batch_size, num_points, 3)
            
            feature_transform, outputs = model(pos_reshaped)
            preds = outputs.max(1)[1]
            all_predictions.append(preds.cpu())
    return torch.cat(all_predictions, dim=0)


def predict_segmentation(model, data_loader, num_classes):
    """
    Predict segmentation labels for each point in the point cloud.
    
    Args:
        model: PointNet segmentation model
        data_loader: Test data loader
        num_classes: Number of segmentation classes
    
    Returns:
        all_predictions: Predicted labels for all points
        all_targets: Ground truth labels for all points
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for pointcloud, pc_class, label, seg_class in data_loader:
            pointcloud = pointcloud.to(device)
            seg_class = seg_class.to(device)
            
            # Transpose from (batch, 3, num_points) to (batch, num_points, 3)
            pointcloud = pointcloud.transpose(2, 1)
            
            # Forward pass
            feature_transform, predictions = model(pointcloud)
            
            # Reshape predictions and labels
            # predictions: (batch_size, num_classes, num_points)
            # seg_class: (batch_size, num_points)
            predictions = predictions.transpose(2, 1).contiguous()
            predictions = predictions.view(-1, num_classes)
            seg_class = seg_class.view(-1)
            
            # Get predicted labels
            pred_labels = predictions.argmax(dim=1)
            
            all_predictions.append(pred_labels.cpu())
            all_targets.append(seg_class.cpu())
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    return all_predictions, all_targets


def evaluate_segmentation(model, data_loader, num_classes):
    """
    Evaluate segmentation model and compute metrics.
    
    Args:
        model: PointNet segmentation model
        data_loader: Test data loader
        num_classes: Number of segmentation classes
    
    Returns:
        accuracy: Overall point-wise accuracy
        mean_iou: Mean IoU across all classes
    """
    from train import calculate_iou
    
    predictions, targets = predict_segmentation(model, data_loader, num_classes)
    
    # Calculate accuracy
    accuracy = (predictions == targets).sum().item() / targets.size(0)
    
    # Calculate IoU
    mean_iou = calculate_iou(predictions, targets, num_classes)
    
    return accuracy, mean_iou