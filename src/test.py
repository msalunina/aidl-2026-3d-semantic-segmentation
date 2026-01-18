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