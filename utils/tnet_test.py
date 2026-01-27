import torch
from torch.utils.data import DataLoader

# from utils.shapenet_dataset import shapeNetDataset  # if you run from repo root
from shapenet_dataset import shapeNetDataset      # if you run from inside utils

import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """Learns a (B, dim, dim) transform. Input is (B, dim, N)."""

    def __init__(self, dim: int, num_points: int):
        super().__init__()
        self.dim = dim
        self.num_points = num_points

        self.conv1 = nn.Conv1d(dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, dim * dim)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.maxpool = nn.MaxPool1d(kernel_size=num_points)

        # start close to identity
        nn.init.constant_(self.fc3.weight, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, dim, N)
        B, D, N = x.shape
        assert D == self.dim, f"Expected dim={self.dim}, got {D}"
        assert N == self.num_points, f"Expected N={self.num_points}, got {N}"

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.maxpool(x).squeeze(-1)  # (B, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # (B, dim*dim)

        I = torch.eye(self.dim, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)
        x = x.view(B, self.dim, self.dim) + I
        return x


def apply_transform(x: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """
    x: (B, 3, N)
    trans: (B, 3, 3)
    returns: (B, 3, N)
    """
    return torch.bmm(trans, x)


def main():
    config = {
        "dataset_path": r"C:\aidl\Project\shapenet\PartAnnotation",
        "num_points": 1024,
        "batch_size": 1,
    }

    dataset = shapeNetDataset(config["dataset_path"], config["num_points"], mode=0, class_name="")
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tnet = TNet(dim=3, num_points=config["num_points"]).to(device)
    tnet.eval()

    # Grab one batch
    pointcloud, obj_class, part_labels, seg_class = next(iter(loader))

    # NEW DATASET: pointcloud is already (B, 3, N)
    x = pointcloud.float().to(device)

    with torch.no_grad():
        trans = tnet(x)               # (B, 3, 3)
        x_aligned = apply_transform(x, trans)

    print("Input x:", x.shape)
    print("Transform:", trans.shape)
    print("Aligned x:", x_aligned.shape)

    det = torch.det(trans).cpu()
    print("det(transform) per sample:", det.tolist())

    I = torch.eye(3).unsqueeze(0).repeat(trans.shape[0], 1, 1).to(device)
    ortho_err = torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)).cpu()
    print("||T T^T - I|| per sample:", ortho_err.tolist())

    ident_err = torch.norm(trans - I, dim=(1, 2)).cpu()
    print("||T - I|| per sample:", ident_err.tolist())


if __name__ == "__main__":
    main()
