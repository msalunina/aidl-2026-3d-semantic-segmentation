import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader as DataLoaderTorch   
from utils_training import eval_single_epoch_segmentation, set_seed
from utils_data import load_dataset, choose_architecture, info_dataset_batch

def main():

    # FORCE CPU
    device = torch.device("cpu")

    # LOAD TRAINED STATE
    load_path = "checkpoints/SegPointNet_Airplane_ShapeNet_1024pts_1epochs.pt"    # includes config    
    checkpoint_state = torch.load(load_path, map_location=device)
    config = checkpoint_state["config"]  
    hyper  = checkpoint_state["hyper"]
    
    # SEEDS i histories varies
    set_seed(config["seed"])

    # DATASET 
    _, _, test_dataset, id_to_name = load_dataset(config)
    num_classes = len(id_to_name)

    # DATALOADER
    test_loader  = DataLoaderTorch(test_dataset, batch_size=hyper["batch_size"], shuffle=False)

    # CHOOSE AND UPDATE ARCHITECTURE
    network_trained = choose_architecture(config["architecture"], num_classes)
    network_trained.load_state_dict(checkpoint_state["model"])
    network_trained.eval()      # changes behaviour of some layers (e.g. dropout off), does not strop gradient

    # /////////////////////////////////////////////////////////////////// 
    # /////////////////////////// FINAL TEST ////////////////////////////
    # /////////////////////////////////////////////////////////////////// 
    criterion = nn.NLLLoss(ignore_index=-1)
    with torch.no_grad():       # Stops tracking gradients, saves memory
        test_loss, test_acc = eval_single_epoch_segmentation(test_loader, network_trained, criterion)
        print(f"Final test: loss={test_loss:.4f}, acc={test_acc:.2f}")  
    # /////////////////////////////////////////////////////////////////// 

if __name__ == "__main__":
    main()