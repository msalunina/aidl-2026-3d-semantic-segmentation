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
