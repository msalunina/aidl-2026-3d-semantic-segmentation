import laspy
import torch
import os
from pathlib import Path
import numpy as np

class lasDataset(torch.utils.data.Dataset):

    def __init__(self, files_path: str):
        super().__init__()
        self.files_path = files_path
        self.getLasFiles()

    def getLasFiles(self):
        self.files = []
        data_dir = Path(self.files_path)
        #Get the list of las files in folder
        las_files = list(data_dir.glob('*.las'))
        print(f"Found {len(las_files)} LAS files:")
        for f in las_files:
            print(f"  - {f.name}")
            self.files.append(f.name)

    def __len__(self) ->int:
        return len(self.files)
        
    def __getitem__(self, index:int):
        las_file = os.path.join(self.files_path, self.files[index])
        las_data = laspy.read(las_file)
        xyz = np.vstack((las_data.x, las_data.y, las_data.z)).T
        
        #create a tensor with xyz values
        #return the labels
        xyz_tensor = torch.from_numpy(xyz).float()
        return xyz_tensor
        #get number of points
        

