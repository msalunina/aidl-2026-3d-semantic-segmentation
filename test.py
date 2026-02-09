import torch
import torch.nn as nn
from pathlib import Path
from torch_geometric.loader import DataLoader as DataLoaderGeometric
from torch.utils.data import DataLoader as DataLoaderTorch   
from utils_training import eval_single_epoch, eval_single_epoch_segmentation, set_seed
from utils_data import load_dataset, choose_architecture, info_dataset_batch

def main():

    # FORCE CPU
    device = torch.device("cpu")

    # LOAD TRAINED STATE
    # RUN_NAME = "ClassPointNet_ShapeNet_1024pts_1epochs"
    # RUN_NAME = "ClassPointNet_ModelNet_1024pts_10epochs"
    # RUN_NAME = "ClassPointNetSmall_ShapeNet_1024pts_1epochs"
    # RUN_NAME = "ClassPointNetSmall_ModelNet_1024pts_30epochs"
    RUN_NAME = "SegPointNet_Airplane_ShapeNet_1024pts_10epochs"
    run_dir = Path("runs") / RUN_NAME
    checkpoint_path = run_dir / "checkpoint.pt"

    checkpoint_state = torch.load(checkpoint_path, map_location=device, weights_only=False)    # it loads more things that weights
    config       = checkpoint_state["config"] 
    dataset      = checkpoint_state["config"]["dataset"]  
    architecture = checkpoint_state["config"]["architecture"]  
    batch_size = 32
    
    # SEEDS i histories varies
    set_seed(config["seed"] )

    # DATASET 
    _, _, test_dataset, id_to_name = load_dataset(config)
    num_classes = len(id_to_name)

    # DATALOADER
    if dataset == "ModelNet":
        test_loader  = DataLoaderGeometric(test_dataset, batch_size=batch_size, shuffle=False)
    elif dataset == "ShapeNet":
        test_loader  = DataLoaderTorch(test_dataset, batch_size=batch_size, shuffle=False)
    else: raise TypeError(f"{dataset} is not an option") 

    # CHOOSE AND UPDATE ARCHITECTURE
    network_trained = choose_architecture(architecture, num_classes)
    network_trained.load_state_dict(checkpoint_state["model"])
    network_trained.eval()      # changes behaviour of some layers (e.g. dropout off), does not strop gradient

    # /////////////////////////////////////////////////////////////////// 
    # /////////////////////////// FINAL TEST ////////////////////////////
    # /////////////////////////////////////////////////////////////////// 
    if architecture in {"ClassPointNet", "ClassPointNetSmall"}:
        criterion = nn.NLLLoss()
        with torch.no_grad():       # Stops tracking gradients, saves memory
            test_loss, test_acc = eval_single_epoch(test_loader, network_trained, criterion)
            print(f"Final test: loss={test_loss:.4f}, acc={test_acc:.2f}")

    elif architecture == "SegPointNet":
        criterion = nn.NLLLoss(ignore_index=-1)
        with torch.no_grad():       # Stops tracking gradients, saves memory
            test_loss, test_acc, test_miou = eval_single_epoch_segmentation(test_loader, network_trained, criterion)
            print(f"Final test: loss={test_loss:.4f}, acc={test_acc:.2f}, miou={test_miou:.2f}")  

    else: raise TypeError(f"{architecture} is not an option") 
   
    # /////////////////////////////////////////////////////////////////// 

if __name__ == "__main__":
    main()