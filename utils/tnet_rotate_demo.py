import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from shapenet_dataset import shapeNetDataset  # script lives inside utils/


# -----------------------------
# T-Net (same as in tnet_test.py)
# -----------------------------
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
    Accepts x as (B,3,N) or (B,N,3). Returns (B,3,N).
    trans: (B,3,3)
    """
    if x.ndim != 3:
        raise ValueError(f"x must be 3D, got shape {tuple(x.shape)}")

    # If x is (B,N,3), convert to (B,3,N)
    if x.shape[1] != 3 and x.shape[2] == 3:
        x = x.permute(0, 2, 1).contiguous()

    # Now must be (B,3,N)
    if x.shape[1] != 3:
        raise ValueError(f"Expected x to be (B,3,N) after fixup, got {tuple(x.shape)}")

    return torch.bmm(trans, x)


# -----------------------------
# Rotation helper (known rotation for testing)
# -----------------------------
def rotation_z(deg: float, device, dtype):
    rad = deg * math.pi / 180.0
    c = math.cos(rad)
    s = math.sin(rad)
    R = torch.tensor(
        [[c, -s, 0.0],
         [s,  c, 0.0],
         [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype
    )
    return R


# -----------------------------
# Plotting helper
# -----------------------------
def plot_three(title, A, B, C):
    """
    A/B/C are (N,3) numpy arrays.
    """
    fig = plt.figure(figsize=(12, 4))
    fig.suptitle(title)

    for idx, (pts, name) in enumerate([(A, "Original"),
                                       (B, "Rotated input"),
                                       (C, "T-Net undo (aligned)")], start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2)
        ax.set_title(name)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

        # keep same viewing limits for fair comparison
        lim = 0.6
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)

    plt.tight_layout()
    plt.show()


def main():
    # ---- config ----
    config = {
        "dataset_path": r"C:\aidl\Project\shapenet\PartAnnotation",
        "num_points": 1024,
        "batch_size": 4,
        "rotation_deg": 45.0,
        "steps": 300,
        "lr": 1e-3,
    }

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load batch from ShapeNet ----
    dataset = shapeNetDataset(config["dataset_path"], config["num_points"], mode=0, class_name="")
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)

    pointcloud, obj_class, part_labels, seg_class = next(iter(loader))

    # NEW shapenet_dataset.py should already output (B,3,N) float32-ish,
    # but we handle both cases safely:
    x_orig = pointcloud.float().to(device)
    if x_orig.shape[1] != 3 and x_orig.shape[2] == 3:
        x_orig = x_orig.permute(0, 2, 1).contiguous()  # (B,3,N)

    # ---- rotate the input with a known rotation ----
    R = rotation_z(config["rotation_deg"], device=device, dtype=x_orig.dtype)  # (3,3)
    Rb = R.unsqueeze(0).repeat(x_orig.shape[0], 1, 1)  # (B,3,3)
    x_rot = apply_transform(x_orig, Rb)  # (B,3,N)

    # ---- init T-Net (starts at Identity) ----
    tnet = TNet(dim=3, num_points=config["num_points"]).to(device)
    tnet.train()
    opt = torch.optim.Adam(tnet.parameters(), lr=config["lr"])

    # ---- BEFORE training: undo==rotated (since T starts as I) ----
    with torch.no_grad():
        T0 = tnet(x_rot)
        x_undo0 = apply_transform(x_rot, T0)

    A = x_orig[0].transpose(0, 1).detach().cpu().numpy()   # (N,3)
    B = x_rot[0].transpose(0, 1).detach().cpu().numpy()
    C = x_undo0[0].transpose(0, 1).detach().cpu().numpy()

    plot_three(
        title=f"BEFORE training (T-Net starts as Identity) | rotation={config['rotation_deg']}°",
        A=A, B=B, C=C
    )

    # ---- train T-Net to undo the rotation ----
    # Goal: T(x_rot) * x_rot ≈ x_orig
    for step in range(1, config["steps"] + 1):
        opt.zero_grad()

        T = tnet(x_rot)
        x_undo = apply_transform(x_rot, T)

        loss = F.mse_loss(x_undo, x_orig)
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                R_inv = R.t().unsqueeze(0)  # (1,3,3)
                t_err = torch.norm(T - R_inv).item()
            print(f"step {step:4d} | loss={loss.item():.6f} | ||T - R_inv||={t_err:.4f}")

    # ---- AFTER training ----
    tnet.eval()
    with torch.no_grad():
        T1 = tnet(x_rot)
        x_undo1 = apply_transform(x_rot, T1)

    C2 = x_undo1[0].transpose(0, 1).detach().cpu().numpy()

    plot_three(
        title=f"AFTER training (T-Net learned to undo rotation) | rotation={config['rotation_deg']}°",
        A=A, B=B, C=C2
    )

    print("\nLearned T (approx inverse rotation):")
    print(T1[0].detach().cpu().numpy())
    print("\nTrue inverse rotation R^T:")
    print(R.t().detach().cpu().numpy())


if __name__ == "__main__":
    main()
