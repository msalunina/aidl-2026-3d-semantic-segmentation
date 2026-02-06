import torch

device = torch.device("cpu")

load_path = "checkpoints/ClassPointNetSmall_ShapeNet_1024pts_1epochs.pt"         # includes config
checkpoint_state = torch.load(load_path, map_location=device)
config = checkpoint_state["config"]  
hyper = checkpoint_state["hyper"] 
state_dict = checkpoint_state["model"]

config["architecture"] = "ClassPointNetSmall"

torch.save({"model": state_dict,
            "config": config, 
            "hyper": hyper,
            }, load_path)