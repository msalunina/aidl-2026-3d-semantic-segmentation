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

    #start = time.perf_counter()
    for i, (pointcloud, pc_class, label, seg_class) in enumerate(loader):
     
        #end = time.perf_counter()
        #print(f"getting data elapsed time {end-start:6f} seconds")

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
        #end = time.time()
        #print(f"Training loop time: {end - start:.4f} segundos")
        if(i % config["log_interval"] == 0):
            print(f"train step loss {loss:.5f} acc {accuracy:.5f} total {((i/data_size)*100.0):.2f}%")
        
        #start = time.perf_counter()

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
                print(f"Eval step loss {loss:.5f} acc {accuracy:.5f} total {((i/data_size)*100.0):.2f}%")

    
    return np.mean(losses), np.mean(accs)

def train_shapenet_dataset(config):
    
    batch_size = config["batch_size"]
    writer = SummaryWriter(log_dir="./runs/pointnet_class_norm_25ep")

    model = ClassificationPointNet(num_classes=config["classes"],
                                   point_dimension=3)
    
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

    start_train = time.perf_counter()

    for epoch in range(config["epochs"]):

        start = time.perf_counter()
        loss, acc = train_single_epoch(model, train_loader, optimizer, config, writer)
        end = time.perf_counter()
        print(f"Train Epoch {epoch} loss={loss:.2f} acc={acc:.2f} elapsed time {end-start:6f} seconds")
        
        writer.add_scalar(tag="Shapenet_Train/Loss", scalar_value=loss, global_step=epoch)
        writer.add_scalar(tag="Shapenet_Train/Accuracy", scalar_value=acc, global_step=epoch)
        
        start = time.perf_counter()
        loss, acc = eval_single_epoch(model, eval_loader, config, writer)
        end = time.perf_counter()
        
        print(f"Eval Epoch {epoch} loss={loss:.2f} acc={acc:.2f} elapsed time {end-start:6f} seconds")

        writer.add_scalar(tag="Shapenet_validation/Loss", scalar_value=loss, global_step=epoch)
        writer.add_scalar(tag="Shapenet_validation/Accuracy", scalar_value=acc, global_step=epoch)

        if(epoch % config["checkpoint"] == 0):
            savedir = f"snaps/pointnet_shapenet_{epoch+1}_epochs.pt"
            
            print(f"Saving checkpoint to {savedir}...")

            checkpoint = {
                "model_state_dict": model.cpu().state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_class": config["classes"],
            }
    
            torch.save(checkpoint, savedir)
            
        #put the model again in GPU after saving snapshot
        model.to(device)
      
    #save the trained version
    end_train = time.perf_counter()
    print(f"Training finished, total time {end_train -start_train:.6f} seconds") 
    savedir = f"snaps/pointnet_shapenet_{epoch+1}_epochs.pt"
            
    print(f"Saving checkpoint to {savedir}...")

    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "num_class": config["classes"],
    }
    
    torch.save(checkpoint, savedir)
    #loss, acc = eval_single_epoch(model, test_loader)
    #print(f"Test loss={loss:.2f} acc={acc:.2f}")
    writer.close()


if __name__ == "__main__":

    config = {
        #"dataset_path": "/mnt/456c90d8-963b-4daa-a98b-64d03c08e3e1/Black_1TB/datasets/shapenet/PartAnnotation",
        "dataset_path": "F:/AIDL_FP/Datasets/PartAnnotation",
        "point_cloud_size": 1024,
        "epochs": 25,
        "lr": 1e-3,
        "log_interval": 40,
        "batch_size": 2,
        "classes": 16,
        "checkpoint": 5
    }

    train_shapenet_dataset(config)