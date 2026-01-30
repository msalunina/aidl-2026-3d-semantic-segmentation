import torch
import torch.nn as nn
from PointNet import ClassificationPointNet
import numpy as np
import random
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from training_utils import eval_single_epoch


 # SEEDS i histories varies
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# FORCE CPU
device = torch.device("cpu")

def main():
    config = {"epochs": 10,
                "lr": 0.001,
                "batch_size": 32,
                "nPoints": 1024} 

    # TRANSFORMS
    transform = T.Compose([
        T.SamplePoints(config["nPoints"]),      # mesh -> point cloud (pos: [N,3])
        T.NormalizeScale(),                     # center + scale to unit sphere
    ])

    # DATASET + DATALOADER
    # "pre_transform" is processed only once and saved to processed/*.pt forever (until deleted)
    # "transform" is processed every time __getitem__ is called, every time an object is called (i.e. it samples it)
    test_dataset  = ModelNet(root="data/ModelNet", name="10", train=False, pre_transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False) 

    # LOAD TRAINED NETWORK
    model_path = "checkpoints/ClassificationPointnet_10epochs_1024.pt"
    checkpoint_state = torch.load(model_path, map_location=device)
    network_trained = ClassificationPointNet(num_classes=10)
    network_trained.load_state_dict(checkpoint_state["model"])
    network_trained.eval()      # changes behaviour of some layers (e.g. dropout off), does not strop gradient

    # FINAL TEST
    criterion = nn.NLLLoss()
    with torch.no_grad():       # Stops tracking gradients, saves memory
        test_loss, test_acc = eval_single_epoch(test_loader, network_trained, criterion)
        print(f"Final test: loss={test_loss:.4f}, acc={test_acc:.2f}")


if __name__ == "__main__":
    main()