import numpy as np
import torch
import json
from pathlib import Path
from torch.utils.data import Dataset
import os
import random

class shapenetDataset(Dataset):

    def __init__(self, path:str, mode:str, pc_size:int):
        self.path = path #this has to be the path to the raw data
        self.mode = mode
        self.pc_size = pc_size

        self.data = []
        self.loadData()


    def loadData(self):
        
        file_split = "shuffled_train_file_list.json"

        if(self.mode == "eval"):
            file_split = "shuffled_val_file_list.json"
        elif(self.mode == "test"):
            file_split = "shuffled_test_file_list.json"

        split = os.path.join(self.path, "train_test_split", file_split)
        #read the json file

        with open(split, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        
    def processData(self, pc, labels):
        len_pc = len(pc)

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

        return pc, labels

    def __getitem__(self, index):
        
        file_path = self.data[index]
        file_path = file_path.replace("shape_data", self.path) + "_8x8.npz"
        data = np.load(file_path)
        points = data['pc'] # N,3
        labels = data['part_label'] 
        points, labels = self.processData(points, labels) #downsample the data
        return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int64)
        
    
    def __len__(self):
        return len(self.data)


