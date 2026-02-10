from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader

batch_size = 1

full_train_dataset = ModelNet(root="data/ModelNet", name="10", train=True)
train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True) 

for data in train_loader:
    print("loop")
