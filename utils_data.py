import os
from torch_geometric.utils import to_dense_batch
import numpy as np

# ----------------------------------------------------
#         LOAD DATASET, LOADER AND NUMA_CLASSES
# ----------------------------------------------------
def load_dataset(config):
    """
    Load datasets and class/part names according to the experiment configuration.

    Parameters
    ----------
    config : dict
        Must contain:
        - "dataset"  : {"ModelNet", "ShapeNet"}
        - "nPoints"  : int

        ShapeNet only:
        - "class_name" : str
            ""   → load all object classes (classification)
            name → load a single object class (segmentation)

    Returns
    -------
    train_dataset : Dataset
    val_dataset   : Dataset or None
    test_dataset  : Dataset
    id_to_name    : dict[int, str]
        Maps output label ids to names:
        - ModelNet → object class names
        - ShapeNet + class_name == "" → object class names
        - ShapeNet + class_name != "" → part names for the selected object
    """

    # ////////////////////////////////
    #             MODELNET 
    # ////////////////////////////////   
    if config["dataset"] == "ModelNet":    
        import torch_geometric.transforms as T
        from torch_geometric.datasets import ModelNet
        data_path = "data/ModelNet"
        
        # Importing ModelNet
        # ModelNet dataset (original) stores objects as triangular meshes (.off files).
        # PointNet needs points, not meshes. "SamplePoints" (from torch_geometric) does that: 
        # samples N unifrom points from the object surface and gives you back a tensor
        transform = T.Compose([T.SamplePoints(config["nPoints"]),      # mesh -> point cloud (pos: [N,3])
                               T.NormalizeScale()])                    # center + scale to unit sphere
        # "pre_transform" is processed only once and saved to processed/*.pt forever (until deleted)
        # "transform" is processed every time __getitem__ is called, every time an object is called (i.e. it samples it)
        train_dataset = ModelNet(root=data_path, name="10", train=True, pre_transform=transform)    
        test_dataset = ModelNet(root=data_path, name="10", train=False, pre_transform=transform)    

        # GET NAME CLASSES AND CREATE DICTIONARY
        # listdir: lists everything inside a directory
        # isdir: says if it is a directory
        folder_path = os.path.join(data_path, "raw")
        name_classes = sorted( d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d)))
        id_to_name = {i: n for i, n in enumerate(name_classes)}

        return train_dataset, None, test_dataset, id_to_name
    
    # ////////////////////////////////
    #             SHAPENET 
    # ////////////////////////////////
    elif config["dataset"] == "ShapeNet":
        from utils.shapenet_dataset_fast import shapeNetDataset
        data_path = "data/ShapeNet/PartAnnotation"

        class_name = config["class_name"]  
     
        train_dataset = shapeNetDataset(dataset_path=data_path, point_cloud_size=config["nPoints"], mode=0, class_name=class_name)
        val_dataset = shapeNetDataset(dataset_path=data_path, point_cloud_size=config["nPoints"], mode=1, class_name=class_name)
        test_dataset = shapeNetDataset(dataset_path=data_path, point_cloud_size=config["nPoints"], mode=2, class_name=class_name)
        assert class_name == "" or class_name in train_dataset._metadata, f"Unknown ShapeNet class_name: {class_name}"

        if class_name == "":
            # Classification or multi-class setup --> return object classes
            object_classes = train_dataset._object_classes              # DICTIONARY name -> id
            id_to_name = {v: k for k, v in object_classes.items()}      # CREATES A DICTIONARY id -> name (reverse order)
            name_classes = [id_to_name[i] for i in range(len(id_to_name))]

        else:
            # Segmentation on a single object --> return part names
            name_classes = train_dataset._metadata[class_name]["lables"] 
            id_to_name = {i: n for i, n in enumerate(name_classes)}                 # CREATES A DICTIONARY


        return train_dataset, val_dataset, test_dataset, id_to_name
    
    else:
        raise TypeError(f"No idea what is dataset {config['dataset']}")
    


# ----------------------------------------------------
#       CHOOSE ARCHITECTURE (NETWORK)
# ----------------------------------------------------
def choose_architecture(architecture, num_classes):

    if architecture == "ClassPointNetSmall": 
        from PointNetSmall import ClassificationPointNet as ClassificationPointNetSmall
        network = ClassificationPointNetSmall(num_classes=num_classes)

    elif architecture == "ClassPointNet":  
        from PointNet import ClassificationPointNet
        network = ClassificationPointNet(num_classes=num_classes)

    elif architecture == "SegPointNet":
        from PointNet import SegmentationPointNet 
        network = SegmentationPointNet(num_classes=num_classes)

    else:
        raise TypeError(f"No idea what is architecture {architecture}")

    return network



def info_dataset_batch(dataset_name, dataset, loader, id_to_name=None):

    # ////////////////////////////////
    #             MODELNET  
    # ////////////////////////////////
    if dataset_name == "ModelNet":  
        # There are 32 objects per batch, and each object has N points. 
        # DataLoader glues many objects together, it does not separate objects....all 32xN points are stacked together.
        # batch is a Data object (like a struct in matlab), with attributes. The atributes are:
        # batch.pos   --> llista de punts per batch ([32xN,3])
        # batch.batch --> integer telling each point to which object it belongs to.
        # batch.y     --> labels, one label per object (not point!!!)
        # batch.ptr   --> ni puta idea
        # They can also be accessed like a dictionary: batch["pos"], batch["y"]...
       
        # info item from dataset
        i = 0
        print(f"\ndataset: {dataset_name}")
        print(id_to_name)
        print(f"\ndataset[i]:\n{dataset[i]}")
        print(f"  dataset[i].pos (points):      {dataset[i].pos.shape}")
        print(f"  dataset[i].y   (labels):      {dataset[i].y.shape}    although scalar, 1 dim tensor to facilitate concatenating")

        # info item from a batch
        batch = next(iter(loader))
        print(f"\nbatch:\n{batch}")
        print(f"  batch.pos (points batch):     {batch.pos.shape} = (objectsxN,3) --> it has to be split")
        print(f"  batch.y   (labels batch):     {batch.y.shape}")

        print(f"\nbatch[i]: DOES NOT GIVE YOU OBJECT i!!! to get object i:")
        i = 0
        points = batch.pos[batch.batch == i]
        label = batch.y[i]
        print(f"  Points object i: batch.pos[batch.batch==i]    {points.shape}")
        print(f"  Label  object i: batch.y[i]                   {label.shape}  elem from a 1d tensor --> scalar")

        print(f"OR using:\npoints_all_objects, _ = to_dense_batch(batch.pos, batch.batch)")
        points_all_objects, _ = to_dense_batch(batch.pos, batch.batch)
        print(f"points_all_objects:     {points_all_objects.shape}")
        print(f"points_all_objects[i]:  {points_all_objects[i].shape}")


    # ////////////////////////////////
    #             SHAPENET 
    # ////////////////////////////////
    elif dataset_name == "ShapeNet": 
        # dataset[sample] -> (points, object_class, seg_labels, num_seg_classes)

        print(f"\ndataset: {dataset_name}")
        print(id_to_name)
        print(f"\ndataset[i]")
        sample = 800
        points, object_class, seg_labels, global_labels = dataset[sample]
        print(f"  Points:               {points.shape}")
        print(f"  Object class:         {object_class}")
        print(f"  Seg labels:           {seg_labels.shape} --> {np.unique(seg_labels)}")
        print(f"  Global labels:        {global_labels.shape} --> {np.unique(global_labels)}")

        # ---- batch info ----
        batch = next(iter(loader))
        b_points, b_obj, b_seg, b_global = batch

        print("\nbatch (ShapeNet)")
        print(f"  Points batch:            {b_points.shape} ")
        print(f"  Object class batch:      {b_obj.shape}")
        print(f"  Seg labels batch:        {b_seg.shape}")
        print(f"  Global labels batch:     {b_global.shape}")




