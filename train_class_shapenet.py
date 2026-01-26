import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pointnet import ClassificationPointNet
import numpy as np


from utils.shapenet_dataset import shapeNetDataset
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(model, loader, optimizer, batch_size):
    model.train()
    accs, losses = [], []
    
    data_size = len(loader)
    log_perc = 0.2
    cnt = 0
    for pointcloud, pc_class, label, seg_class in loader:
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
        cnt+=1
        if(cnt % 100):
            print(f"train step loss {loss} acc {accuracy}")
        
    
    return np.mean(losses), np.mean(accs)
    
    
def eval_single_epoch(model, loader):
    accs, losses = [], []
    with torch.no_grad():
        model.eval()
    pass

def train_shapenet_dataset(config):
    
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
    
    
    for epoch in range(config["epochs"]):
        loss, acc = train_single_epoch(model, train_loader, optimizer, config["batch_size"])
        print(f"Train Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
        #loss, acc = eval_single_epoch(model, eval_loader)
        #print(f"Eval Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
    
    loss, acc = eval_single_epoch(model, test_loader)
    print(f"Test loss={loss:.2f} acc={acc:.2f}")

    savedir = "pointnet_shapenet.pt"
    
    print(f"Saving checkpoint to {savedir}...")
    
    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "num_class": config["classes"],
    }
    
    torch.save(checkpoint, savedir)

if __name__ == "__main__":

    config = {
        "dataset_path": "/mnt/456c90d8-963b-4daa-a98b-64d03c08e3e1/Black_1TB/datasets/shapenet/PartAnnotation",
        "point_cloud_size": 1024,
        "epochs": 1,
        "lr": 1e-3,
        "log_interval": 1000,
        "batch_size": 8,
        "classes": 16
    }

    train_shapenet_dataset(config)