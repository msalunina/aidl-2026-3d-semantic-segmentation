import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader   # <-- instead of torch_geometric.loader
from utils_training import train_model_segmentation, plot_curves, set_seed
from utils_data import load_dataset, info_dataset_batch, choose_architecture


def plot_object_parts(dataset, sample):
    
    points, object_class, seg_labels, global_labels = dataset[sample]

    num_parts = len(id_to_name)
    name_parts = [id_to_name[i] for i in range(num_parts)]
    parts_present = len(np.unique(seg_labels))       # equivalent to num_seg_labels but safer....
    # MAPPING PART COLORS
    seg_class_color = ["red", "blue", "green", "black", "orange"]
    points_color = [seg_class_color[i] for i in seg_labels]

    points = points.T

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(points[:,0], points[:,1], points[:,2], s=2)
    ax1.set_title(f"Sample class: {object_class} / Num present parts: {parts_present}/{num_parts}")
    ax1.set_box_aspect([1,1,1])
   
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(points[:,0], points[:,1], points[:,2], c=points_color)
    ax2.set_title(f"Sample parts colored\n{seg_class_color[0:num_parts]}\n{name_parts}")
    ax2.set_box_aspect([1,1,1])
    plt.show()



# RUN ONLY IF EXECUTED AS MAIN
if __name__ == "__main__":

    # Cuda agnostic thingy
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config = {"architecture": "SegPointNet",
              "dataset": "ShapeNet",
              "class_name": "Airplane",         # All classes: "" | Specific class: "Airplane", "Chair"...
              "nPoints": 1024,
              "seed": 42}  
    
    hyper = {"batch_size": 32,
             "epochs": 10,
             "lr": 0.001}
    
    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/{config['architecture']}_{config['class_name']}_{config['dataset']}_{config['nPoints']}pts_{hyper['epochs']}epochs.pt"
    print(save_path)

    # SEEDS i histories varies
    set_seed(config["seed"])

    # LOADING SHAPENET DATASET
    train_dataset, val_dataset, test_dataset, id_to_name = load_dataset(config)

    # DATALOADERS
    train_loader = DataLoader(train_dataset, batch_size=hyper["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=hyper["batch_size"], shuffle=False)
    
    # CHECKS: sizes
    # info_dataset_batch(config["dataset"], train_dataset, train_loader, id_to_name)
    print(f"\nTrain size: {len(train_dataset)}")
    print(f"Val size  : {len(val_dataset)}")
    print(f"Test size : {len(test_dataset)}")
    print(f"Class name: {config['class_name']}")
    print(f"id to name: {id_to_name}\n")  

    # CHOOSE: multi-class segmentation or single-class segmentation
    if config["class_name"] == "":
        global_id_to_name = train_dataset.global_id_to_name
        num_classes = len(global_id_to_name)
    else:
        num_classes = len(id_to_name)


    plot_object_parts(train_dataset, 0)
 
    # MODEL + OPTIMIZER + LOSS
    network = choose_architecture(config["architecture"], num_classes).to(device)
    optimizer = optim.Adam(network.parameters(), lr=hyper["lr"])
    criterion = nn.NLLLoss(ignore_index=-1)         # when label is -1 (ground truth) do not include it in teh loss

    # TRAINING LOOP
    time_start_traning = time.time()
    train_loss, train_acc, val_loss, val_acc = train_model_segmentation(hyper, train_loader, val_loader, network, optimizer, criterion)
    time_training = time.time() - time_start_traning
    print(f"Training time: {time_training}")

    torch.save({"model": network.state_dict(),
                "config": config, 
                "hyper": hyper,
                }, save_path)

    # PLOTTING CURVES
    plot_curves(train_loss, train_acc, val_loss, val_acc)