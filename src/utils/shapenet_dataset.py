from datetime import datetime
import numpy as np
import torch
import json
from pathlib import Path
from torch.utils.data import Dataset
import os
import random

class shapenetDataset(Dataset):

    category_ids = {
        'Airplane': '02691156',
        'Bag': '02773838',
        'Cap': '02954340',
        'Car': '02958343',
        'Chair': '03001627',
        'Earphone': '03261776',
        'Guitar': '03467517',
        'Knife': '03624134',
        'Lamp': '03636649',
        'Laptop': '03642806',
        'Motorbike': '03790512',
        'Mug': '03797390',
        'Pistol': '03948459',
        'Rocket': '04099429',
        'Skateboard': '04225987',
        'Table': '04379243',
    }

    category_names = {
        '02691156' : 'Airplane',
        '02773838' : 'Bag',
        '02954340' : 'Cap',
        '02958343' : 'Car',
        '03001627' : 'Chair',
        '03261776' : 'Earphone',
        '03467517' : 'Guitar',
        '03624134' : 'Knife',
        '03636649' : 'Lamp',
        '03642806' : 'Laptop',
        '03790512' : 'Motorbike',
        '03797390' : 'Mug',
        '03948459' : 'Pistol',
        '04099429' : 'Rocket',
        '04225987' : 'Skateboard',
        '04379243' : 'Table',
    }

    seg_classes = {
        'Airplane': [0, 1, 2, 3],
        'Bag': [4, 5],
        'Cap': [6, 7],
        'Car': [8, 9, 10, 11],
        'Chair': [12, 13, 14, 15],
        'Earphone': [16, 17, 18],
        'Guitar': [19, 20, 21],
        'Knife': [22, 23],
        'Lamp': [24, 25, 26, 27],
        'Laptop': [28, 29],
        'Motorbike': [30, 31, 32, 33, 34, 35],
        'Mug': [36, 37],
        'Pistol': [38, 39, 40],
        'Rocket': [41, 42, 43],
        'Skateboard': [44, 45, 46],
        'Table': [47, 48, 49],
    }

    def __init__(self, path:str, mode:str, pc_size:int, prob_rotation:float):
        self.path = path #this has to be the path to the raw data
        self.mode = mode
        self.pc_size = pc_size
        self.num_classes = 50
        self.rot_prob = prob_rotation
        self.rotation_deg_max = 180.0

        self.data = []
        self.loadData()

    def num_classes(self):
        return self.num_classes
       
    def loadData(self):
        
        #train by default
        file_split = "shuffled_train_file_list.json"

        if(self.mode == "eval"):
            file_split = "shuffled_val_file_list.json"
        elif(self.mode == "test"):
            file_split = "shuffled_test_file_list.json"

        split = os.path.join(self.path, "train_test_split", file_split)
        #read the json file

        with open(split, "r", encoding="utf-8") as f:
            self.data = json.load(f)
               
       
    def _augment_points(self, points: np.ndarray):
        """
        Apply random geometric augmentation to XYZ coordinates.

        Augmentations:
        - Random rotation around Y-axis
        - Random isotropic scaling
        """
        points = points.copy()

        if points.shape[1] < 3:
            return points

        xyz = points[:, :3]

        angle_rad = np.deg2rad(np.random.uniform(-self.rotation_deg_max, self.rotation_deg_max))
        cos_theta, sin_theta = np.cos(angle_rad), np.sin(angle_rad)
        rotation = np.array([
            [ cos_theta, 0.0, sin_theta],
            [ 0.0,       1.0, 0.0],
            [-sin_theta, 0.0, cos_theta],
        ], dtype=np.float32)

        xyz = np.matmul(xyz, rotation.T)

        points[:, :3] = xyz
        return points
        
    def processData(self, pc, labels):
        len_pc = len(pc)

        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        max_dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        
        if max_dist > 0:
            pc = pc / max_dist

        if(len_pc < self.pc_size): # "interpolate"
            diff = self.pc_size - len_pc
            idx = np.random.choice(len_pc, diff, replace=True)
            ipc = pc[idx]
            il = labels[idx]
            pc = np.concatenate([pc, ipc])
            labels = np.concatenate([labels,il])
        else: #"downsample"
            idx = np.random.choice(len_pc, size=self.pc_size, replace=False)
            #index = random.sample(range(len_pc), self.pc_size)
            pc = pc[idx]
            labels = labels[idx]
        
        #randomly rotate the pointcloud
        rot = (np.random.rand() < self.rot_prob)
        if(rot):
            pc = self._augment_points(pc)
        
        #add gaussian noise mean 0 and 0.02 standar dev
        noise = np.clip(np.random.normal(0.0, 0.02, (self.pc_size,3)), -0.05, 0.05)
        pc = pc+noise
        return pc, labels
    
    def read_txt(self, path):
        pointcloud = []
        labels = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                values = line.split(" ")[:3]
                label = line.split(" ")[-1]
                pointcloud.append(values)
                labels.append(float(label))
        
        return pointcloud,labels
    
    def getPartsInObject(self, labels, parts_labels):

        parts_count = np.zeros(len(parts_labels)) #zero vector with len number of classes in object
        hist = np.bincount(labels,  minlength=50)
        endp = parts_labels[0] + len(parts_labels)
        parts_count = hist[parts_labels[0]: endp]
        return parts_count

    def __getitem__(self, index):
        
        file_path = self.data[index]        
        file_path = file_path.replace("shape_data", self.path) + ".txt" #"_8x8.npz"
        points, label = self.read_txt(file_path)
        points = np.array(points, dtype=np.float32)
        labels = np.array(label, dtype=np.int64)
        
        #get the IoU offset, if a class not in the object add 1/n_class
        #- get the object class
        parts_count = None
        parts_labels = None
        class_object = 0
        for class_id, name in self.category_names.items():
            if class_id in file_path:
                parts_labels = self.seg_classes[name]
                break
            class_object +=1
        #adjust the data to expected size
        points, labels = self.processData(points, labels) 
        #get count of parts for each label
        parts_count = self.getPartsInObject(labels, parts_labels)
        iou_offset = 0
        for i in parts_count:
            if i == 0:
                iou_offset += (1.0/len(parts_count))

        return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int64), np.array(iou_offset, dtype=np.float32), np.array(class_object, dtype=np.int64)
        
    
    def __len__(self):
        return len(self.data)


