import laspy
import torch
import os
from pathlib import Path
import numpy as np
import random

class modelNetDataset(torch.utils.data.Dataset):
    
    def __init__(self, files_path: str, classes_dict: dict, point_cloud_size: int, train: bool):
        super().__init__()
        self.files_path = files_path
        self.classes = classes_dict
        self.train = train
        self.data_files = []
        self.point_cloud_size = point_cloud_size
        self.getFiles()

    def readOffFile(self, file):
        #format file https://segeval.cs.princeton.edu/public/off_format.html
        pointcloud = []

        with open(file, "r") as f:
            line_idx = 0
            num_points = 0
            cur_pt = 0
           
            for line in f:
                if (line_idx == 0):
                    line_idx+=1
                    continue #skip first line
                if(line_idx == 1):
                    line_idx+=1
                    #read the sizes [points][faces][edges]
                    sizes = line.split(" ")
                    num_points = int(sizes[0])
                else:
                    values = line.split(" ")
                    pointcloud.append([float(values[0]), float(values[1]), float(values[2])])
                    line_idx+=1
                    if(len(pointcloud) == num_points):
                        break        
        
        pointcloud = random.sample(pointcloud, self.point_cloud_size)                

        return pointcloud


    def getFiles(self):
        mode = "test"
        if(self.train): mode = "train"
        for class_name in self.classes:
            data_dir = Path(os.path.join(self.files_path,class_name,mode))
            off_files = list(data_dir.glob('*.off'))
            for f in off_files:
                self.data_files.append(os.path.join(self.files_path,class_name,mode,f))


    def __len__(self) ->int:
        return len(self.data_files)
        
    def __getitem__(self, index:int):
       
        off_data = self.readOffFile(self.data_files[index])
        label = None
        for class_name in self.classes:
            if class_name in self.data_files[index]:
                label = self.classes[class_name]
        xyz_tensor = torch.from_numpy(np.array(off_data)).float()
        return xyz_tensor, label
