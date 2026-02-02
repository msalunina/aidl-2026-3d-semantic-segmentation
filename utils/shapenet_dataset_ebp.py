import torch
import os
import json
from pathlib import Path
import numpy as np
import random
import math

class shapeNetDataset(torch.utils.data.Dataset):
    
    #dataset_path: path to the PartAnnotation folder, where the metadata.json is located
    #point_cloud_size: the number of points in the returned point cloud 
    #mode: integer to indicate the mode, train=0,val=1,test=2, by default 0.7 0.2 0.1
    #class_name: single class name for segmentation train in one single class, set "" to use all the classes
    def __init__(self, dataset_path: str, point_cloud_size: int, mode: int, class_name:str):
        super().__init__()
        self._xy_th = 0.05
        self._z_th = 0.05
        self._dataset_path = dataset_path
        self._target_points = point_cloud_size
        self._mode = mode
        self._dataset = []
        self._class_name = class_name

        
        #get the list of files from the folder
        self.loadDataset()
        
        #self._dataset = 
        # [ 
        #       { "points" : "filename",
        #         "labels" : "filename",
        #         "class"  : int, -> index with the object class
        #         "seg_class": int -> number of segmentation class
        #       }
        # ]
    
    def readJsonMetadata(self, metadata_file:str):
        self._object_classes = {}
        self._metadata = {}
        with open(metadata_file, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)
            self._object_classes = { name: i for i, name in enumerate(self._metadata)}
    
    def loadDataset(self):
        
        self.readJsonMetadata(os.path.join(self._dataset_path,"metadata.json"))
        #here in object_classes are the classes with index
        #metadata has all the json file
        
        for class_name in self._object_classes:
            if( (self._class_name) and (self._class_name != class_name)):
                #when only one class is selected, avoid the other classes
                continue 
                
            #get all the files for each class
            class_path = self._metadata[class_name]["directory"]
            
            #point cloud files
            class_dir = Path(os.path.join(self._dataset_path,class_path,"points"))
            point_files = list(class_dir.glob('*.pts'))
            pc_names = [p.name.split(".")[0] for p in point_files]
            
            #label files
            label_dir = Path(os.path.join(self._dataset_path,class_path,"expert_verified","points_label"))
            label_files = list(label_dir.glob('*.seg'))
            
            #the dataset will be always the same, no randomization otherwise may mix data
            tsize = len(label_files)
            
            #compute size for each mode
            train = int(0.7*tsize)
            val = int(0.9*tsize)
            
            #mode labels are stored here
            labels = []
            
            #get the files for each of the modes
            if(self._mode == 0):
                labels = label_files[:train]    
            elif(self._mode == 1):
                labels = label_files[train:val]
            else:
                labels = label_files[val:]
            
            #iterate over the labels
            for file in labels:
                file_name = file.name.split(".")[0]
                #get point cloud for the label
                if(file_name in pc_names):
                #index = pc_names.index(file_name)
                    item = {
                        "points": os.path.join(self._dataset_path,class_path,"points",file_name+".pts"),
                        "labels": os.path.join(self._dataset_path,class_path,"expert_verified","points_label", file_name + ".seg"),
                        "class": self._object_classes[class_name],
                        "seg_class": len(self._metadata[class_name]["lables"]) #There is  a typo in the dataset 
                    }
                    
                    self._dataset.append(item)
          
    #returns if the points are close and part of the same class
    def nearPoint(self, pt1:list, pt2:list, pt1_class:int, pt2_class:int):
        #computes euclidean distance if classes are the same
        nei = False
        if pt1_class != pt2_class:
            return nei

        distxy = math.sqrt(((pt1[0]-pt2[0])**2) + ((pt1[1]-pt2[1])**2))
        distz = abs(pt1[2]-pt2[2])
        nei = ((distxy < self._xy_th) and (distz < self._z_th))
        return nei 
    
    # #downsample the point cloud to have the target size
    # def downsamplePointCloud(self, point_cloud:list, labels:list):
    #     point_cloud_size = len(point_cloud)

    #     #in case the downsample is called wrong, return the same point cloud
    #     if(self._target_points > point_cloud_size):
    #         print(f"Error, cannot downsample {point_cloud_size} to {self._target_points}")
    #         return point_cloud, labels

    #     points_to_remove = point_cloud_size - self._target_points
    #     removed_pos = random.sample(range(0,point_cloud_size), points_to_remove)
        
    #     """
    #     while len(removed_pos) < points_to_remove:
    #         #get a random point    
    #         index = random.randrange(len(point_cloud))    
    #         #continue if the value has been selected
    #         if(index in removed_pos):
    #             continue
    #         #start the searching loop
    #         removed = False
    #         nei = 0
    #         for i in range(len(point_cloud)):
    #             #avoid the same point
    #             if i == index:
    #                 continue
                
    #             if(self.nearPoint(point_cloud[index], point_cloud[i], labels[index], labels[i]) and i not in removed_pos):
    #                 nei+=1
    #             #remove the point that has 2 close neighbors
    #             if(nei >=2):
    #                 removed_pos.append(index)
    #                 removed = True
    #                 break
                
    #         #if the loop has ended, no neighbors have been detected, the point may be a noise, lone point, remove it
    #         if(not removed):
    #             removed_pos.append(index)
    #     """
    #     #remove the positions
    #     downsampled_pc = [v for i, v in enumerate(point_cloud) if i not in removed_pos]
    #     downsampled_labels = [v for i,v in enumerate(labels) if i not in removed_pos]
    #     return downsampled_pc, downsampled_labels

    #downsample the point cloud to have the target size
    def downsamplePointCloud(self, point_cloud:list, labels:list):
        point_cloud_size = len(point_cloud)

        #in case the downsample is called wrong, return the same point cloud
        if(self._target_points > point_cloud_size):
            print(f"Error, cannot downsample {point_cloud_size} to {self._target_points}")
            return point_cloud, labels

        # Use simple random sampling for efficiency (much faster than the neighbor-based approach)
        indices = np.random.choice(point_cloud_size, self._target_points, replace=False)
        downsampled_pc = [point_cloud[i] for i in indices]
        downsampled_labels = [labels[i] for i in indices]
        return downsampled_pc, downsampled_labels



    
    #interpolate the point cloud to have the target size
    def interpolatePointCloud(self, point_cloud:list, labels:list):
        point_cloud_size = len(point_cloud)

        #in case the interpolation is called wrong, return the same point cloud
        if(self._target_points < point_cloud_size):
            print(f"Error, cannot interpolate {point_cloud_size} to {self._target_points}")
            return point_cloud 

        #create a copy to save the final pointcloud
        interpolated_pc = point_cloud
        interpolated_labels = labels
        points_to_add = self._target_points - point_cloud_size
        interpolated_points = []
        interpolated_index = []
        added_labels = []
        added_points = 0
        
        while added_points < points_to_add:
            #if we have interpolated all the points, and more interpolation is required, the loop should be called again
            #concatenate the interpolated points in point_cloud, and continue interpolation with the interpolated pointcloud
            if( len(interpolated_index) == len(interpolated_pc)):
                interpolated_pc.extend(interpolated_points)
                interpolated_labels.extend(added_labels)
                interpolated_points = []
                interpolated_index = []
                added_labels = []

            #get a random point    
            index = random.randrange(len(point_cloud))    

            #continue if the value has been selected, to interpolate along all the points
            if(index in interpolated_index):
                continue
            
            #start the interpolation loop
            for i in range(len(point_cloud)):
                #avoid the same point
                if i == index:
                    continue
                
                if(self.nearPoint(point_cloud[index], point_cloud[i], labels[index], labels[i])):

                    #this will interpolate in all the nearest points not only one
                    point = [point_cloud[index][0] + (self._xy_th/2.0), point_cloud[index][1] + (self._xy_th/2.0), point_cloud[index][2] + (self._z_th/2.0)]
                    interpolated_points.append(point)
                    added_labels.append(labels[i])
                    added_points += 1

                    #adds the index only once, but interpolates for all the closer points
                    if(index not in interpolated_index): 
                        interpolated_index.append(index)    

                    if(added_points == points_to_add):
                        break
                    
        interpolated_pc.extend(interpolated_points)    
        interpolated_labels.extend(added_labels)      
        return interpolated_pc, interpolated_labels
    
    def normalizePointCloud(self, pc):
        l = pc.shape[0]
        centroid = np.zeros(shape=(3,1), dtype=np.float32)
        centroid[0] = np.mean(pc[0])
        centroid[1] = np.mean(pc[1])
        centroid[2] = np.mean(pc[2])
        pc = pc - centroid
        #m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        #pc = pc / m
        return pc
    
    #reads a single point cloud
    def readPointCloud(self, file_name:str, seg_file:str):
        #read line by line the pts file, contains [x y z]
        #for this dataset X->Depth, Y->height, Z->wide
        #if the coordinate system changes between datasets, does this affect the behavior of the network?
        point_cloud = []
        seg_class = []
        with open(file_name, "r") as f:
            for line in f:
                values = line.split(" ")
                point = [float(values[0]), float(values[1]), float(values[2])]
                point_cloud.append(point)

        with open(seg_file, "r") as f:
            for line in f:
                seg_class.append(int(line)-1)

        return point_cloud, seg_class

    def __len__(self) ->int:
        return len(self._dataset)
    
    def __getitem__(self, index):
        
        item = self._dataset[index]
        point_cloud, labels = self.readPointCloud(item["points"], item["labels"])
        
        target_pc = point_cloud
        target_labels = labels
        
        if(len(point_cloud) > self._target_points):
            #downsample
            target_pc, target_labels = self.downsamplePointCloud(point_cloud, labels)
        elif(len(point_cloud) < self._target_points):
            #interpolate
            target_pc, target_labels = self.interpolatePointCloud(point_cloud, labels)        
        
        target_pc = np.array(target_pc, dtype=np.float32)
        target_pc = target_pc.transpose(1, 0) 
        #target_pc = self.normalizePointCloud(target_pc)
        return target_pc, item["class"], np.array(target_labels), item["seg_class"]      
        
        

        