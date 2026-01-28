import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pointnet import ClassificationPointNet
import numpy as np
import time

from torch.utils.tensorboard import SummaryWriter

from utils.shapenet_dataset import shapeNetDataset
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(model, loader, optimizer, config, writer):
    model.train()
    accs, losses = [], []
    data_size = len(loader)
    batch_size = config["batch_size"]

    for i, (pointcloud, pc_class, label, seg_class) in enumerate(loader):
        start = time.time()
        optimizer.zero_grad()
       
        pointcloud, pc_class = pointcloud.to(device), pc_class.to(device)
        
        preds, feature_transform, tnet_out, ix_maxpool = model(pointcloud)
        
        identity = torch.eye(feature_transform.shape[-1])
        if torch.cuda.is_available():
            identity = identity.cuda()
        regularization_loss = torch.norm(
            identity - torch.bmm(feature_transform, feature_transform.transpose(2, 1)))
        
        # Loss
        loss = F.nll_loss(preds, pc_class) + 0.001 * regularization_loss
        losses.append(loss.cpu().item())
        loss.backward()
        optimizer.step()
        preds = preds.data.max(1)[1]
        corrects = preds.eq(pc_class.data).cpu().sum()

        accuracy = corrects.item() / float(batch_size)
        accs.append(accuracy)
        end = time.time()
        print(f"Training loop time: {end - start:.4f} segundos")
        if(i % config["log_interval"] == 0):
            print(f"train step loss {loss} acc {accuracy} total {(((i+1)*batch_size)/(data_size))*100.0}%")
            writer.add_scalar("Train Loss", loss, i)
            writer.add_scalar("Train Accuracy", accuracy, i)
        
    return np.mean(losses), np.mean(accs)
    
    
def eval_single_epoch(model, loader, config, writer):
    accs, losses = [], []
    data_size = len(loader)
    batch_size = config["batch_size"]

    with torch.no_grad():
        model.eval()
        for i, (pointcloud, pc_class, label, seg_class) in enumerate(loader):

            pointcloud, pc_class = pointcloud.to(device), pc_class.to(device)
        
            preds, feature_transform, tnet_out, ix_maxpool = model(pointcloud)

            loss = F.nll_loss(preds, pc_class)
            losses.append(loss.cpu().item())
            preds = preds.data.max(1)[1]
            corrects = preds.eq(pc_class.data).cpu().sum()
            accuracy = corrects.item() / float(batch_size)
            accs.append(accuracy)
            
            if(i % config["log_interval"] == 0):
                print(f"train step loss {loss} acc {accuracy} total {(((i+1)*batch_size)/(data_size))*100.0}%")

    
    return np.mean(losses), np.mean(accs)

def train_shapenet_dataset(config):
    
    batch_size = config["batch_size"]
    writer = SummaryWriter(log_dir="./runs/exp2")

    model = ClassificationPointNet(num_classes=config["classes"],
                                   point_dimension=config["point_cloud_size"])
    
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    train_dataset = shapeNetDataset(config["dataset_path"], config["point_cloud_size"], 0, "")
    eval_dataset = shapeNetDataset(config["dataset_path"], config["point_cloud_size"], 1, "")
    test_dataset = shapeNetDataset(config["dataset_path"], config["point_cloud_size"], 2, "")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])
    
    print(f"dataset info:\n\ttrain {len(train_dataset)} in {batch_size} total of {len(train_dataset)/batch_size} training loops")
    print(f"\tval {len(eval_dataset)} in {batch_size} total of {len(eval_dataset)/batch_size} evaluation loops")
    print(f"\ttest {len(test_dataset)} in {batch_size} total of {len(test_dataset)/batch_size} test loops")


    for epoch in range(config["epochs"]):

        loss, acc = train_single_epoch(model, train_loader, optimizer, config, writer)
        print(f"Train Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
        loss, acc = eval_single_epoch(model, eval_loader, config, writer)
        print(f"Eval Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")

        writer.add_scalar("Validation Loss", loss, epoch)
        writer.add_scalar("Validation Accuracy", acc, epoch)

        if(epoch % config["checkpoint"] == 0):
            savedir = f"pointnet_shapenet_{epoch}_epochs.pt"
            
            print(f"Saving checkpoint to {savedir}...")

            checkpoint = {
                "model_state_dict": model.cpu().state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_class": config["classes"],
            }
    
            torch.save(checkpoint, savedir)
    
    loss, acc = eval_single_epoch(model, test_loader)
    print(f"Test loss={loss:.2f} acc={acc:.2f}")
    writer.close()


if __name__ == "__main__":

    config = {
        #"dataset_path": "/mnt/456c90d8-963b-4daa-a98b-64d03c08e3e1/Black_1TB/datasets/shapenet/PartAnnotation",
        "dataset_path": "F:/AIDL_FP/Datasets/PartAnnotation",
        "point_cloud_size": 1024,
        "epochs": 10,
        "lr": 1e-3,
        "log_interval": 200,
        "batch_size": 64,
        "classes": 16,
        "checkpoint": 5
    }

    train_shapenet_dataset(config)