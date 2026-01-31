import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import json
import os
from pathlib import Path
from shapenet_dataset import shapeNetDataset
from torch.utils.data import DataLoader
import time

#this can be generated reading the json

"""
shapenet_classes = {
    "Airplane":0,
    "Bag":1,
    "Cap":2,
    "Car":3,
    "Chair":4,
    "Earphone":5,
    "Guitar":6,
    "Knife":7,
    "Lamp":8,
    "Laptop":9,
    "Motorbike":10,
    "Mug":11,
    "Pistol":12,
    "Rocket":13,
    "Skateboard":14,
    "Table": 15
}
"""

class_color = ["red", "blue", "green", "black", "orange"]

#search for close points
#inputs 2 points [x,y,z] search in x,y dimension if, neighboor, looks in z dimension, 
#th1: threshold distance in xy dimension
#th2: threshold distance in z dimension
#returns true if points are close 
def nearPoint(pt1:list, pt2: list, pt1_class:int, pt2_class:int, th1: float, th2: float):
    #compute euclidean distance if classes are the same
    nei = False
    if pt1_class != pt2_class:
        return nei
    
    distxy = math.sqrt(((pt1[0]-pt2[0])**2) + ((pt1[1]-pt2[1])**2))
    distz = abs(pt1[2]-pt2[2])
    nei = ((distxy < th1) and (distz < th2))
    return nei    


#donwsample pointcloud
#remove points from the pointcloud, until the size of the pointcloud is target_points
def donwsamplePointCloud(point_cloud:list, labels:list, target_points:int, xy_th:float, z_th:float):
    point_cloud_size = len(point_cloud)

    #in case the downsample is called wrong, return the same pointcloud
    if(target_points > point_cloud_size):
        print(f"Error, cannot downsample {point_cloud_size} to {target_points}")
        return point_cloud 
    
    points_to_remove = point_cloud_size - target_points
    removed_pos = []
    
    while len(removed_pos) < points_to_remove:
        #get a random point    
        index = random.randrange(len(point_cloud))    
        #continue if the value has been selected
        if(index in removed_pos):
            continue
        
        #start the searching loop
        removed = False
        nei = 0
        for i in range(len(point_cloud)):
            #avoid the same point
            if i == index:
                continue
            
            if(nearPoint(point_cloud[index], point_cloud[i], labels[index], labels[i], xy_th, z_th) and i not in removed_pos):
                nei+=1
            #remove the point that has 2 close neighbors
            if(nei >=2):
                removed_pos.append(index)
                removed = True
                break
            
        #if the loop has ended, no neighbors have been detected, the point may be a noise, lone point, remove it
        if(not removed):
            removed_pos.append(index)
    
    #remove the positions
    downsampled_pc = [v for i, v in enumerate(point_cloud) if i not in removed_pos]
    donwsampled_labels = [v for i,v in enumerate(labels) if i not in removed_pos]
    return downsampled_pc, donwsampled_labels
                
#interpolate pointcloud
def interpolatePointcloud(point_cloud:list, labels:list, target_points:int, xy_th:float, z_th:float):
    point_cloud_size = len(point_cloud)

    #in case the interpolation is called wrong, return the same pointcloud
    if(target_points < point_cloud_size):
        print(f"Error, cannot interpolate {point_cloud_size} to {target_points}")
        return point_cloud 
    
    #create a copy to save the final pointcloud
    interpolated_pc = point_cloud
    interpolated_labels = labels
    points_to_add = target_points - point_cloud_size
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
            
            if(nearPoint(point_cloud[index], point_cloud[i], labels[index], labels[i],xy_th, z_th)):
                
                #this will interpolate in all the nearest points not only one
                point = [point_cloud[index][0] + (xy_th/2.0), point_cloud[index][1] + (xy_th/2.0), point_cloud[index][2] + (z_th/2.0)]
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
            
def readJsonMetadata(metadata_file:str):
    object_classes = {}
    with open(metadata_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        object_classes = { name: i for i, name in enumerate(data)}
            
    return data, object_classes 

def readPointCloud(file_name:str, seg_file:str):
    #read line by line the pts file has [x y z]
    #for this dataset X->Depth, Y->height, Z->wide
    #if the coordinate system changes between datasets, does this affect the behaviour of the network?
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

def showPointCloud(path:str):
    
    #create color array
    #read the data
    metadata, object_classes = readJsonMetadata(os.path.join(path,"metadata.json"))
    #lets begin by reading one airplane
    airplane_path = metadata["Airplane"]["directory"]
    data_dir = Path(os.path.join(path,airplane_path,"points"))
    point_files = list(data_dir.glob('*.pts'))
    
    #this file has all the clases in a single seg file, starting by 1
    label_path = Path(os.path.join(path,airplane_path,"expert_verified","points_label"))
    label_files = list(label_path.glob('*.seg'))
        
    #get the filename for pointcloud data
    pc_files = [p.name.split(".")[0] for p in point_files]
        
    target_points = 4048
    #loop over labels, because labels < pointclouds
    
    for file in label_files:
        
        file_name = file.name.split(".")[0]
        #get the pointcloud for the label
        index = pc_files.index(file_name)
        
        point_cloud, seg_labels = readPointCloud(point_files[index], file)
        
        print(f"Pointcloud {file_name} with points {len(point_cloud)}")
        
        x = [p[0] for p in point_cloud]
        y = [p[1] for p in point_cloud]
        z = [p[2] for p in point_cloud]
        color = [class_color[i] for i in seg_labels]
    
        target_pc = point_cloud
        target_labels = seg_labels
        applied = "Target PointCloud"
        
        if(len(point_cloud) > target_points):
            #downsample
            target_pc, target_labels = donwsamplePointCloud(point_cloud, seg_labels, target_points, 0.05, 0.05)
            applied += " - downsampled"
        elif(len(point_cloud) < target_points):
            #interpolate
            target_pc, target_labels = interpolatePointcloud(point_cloud, seg_labels, target_points, 0.05, 0.05)
            applied = " - interpolated"
 
        tx = [p[0] for p in target_pc]
        ty = [p[1] for p in target_pc]
        tz = [p[2] for p in target_pc]
        tcolor = [class_color[i] for i in target_labels]
    
        #show a point cloud
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.scatter(x, y, z, c=color)
        ax.set_title("original pointcloud")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.scatter(tx, ty, tz, c=tcolor)
        ax2.set_title(applied)
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        plt.tight_layout()
        plt.show()
        
        #break
    
    #we need the points, and the labels for each class,

def showBatchPointcloud(pointcloud, labels, obj_class, seg_class):
    
    for i in range(pointcloud.shape[0]):
        print(f"showing {obj_class[i]} with {seg_class[i]} parts")
        x = pointcloud[i][0,:]
        y = pointcloud[i][1,:]
        z = pointcloud[i][2,:]
        color = [class_color[i] for i in labels[i]]

        #show a point cloud
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, c=color)
        ax.set_title("original pointcloud")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    
    
        plt.tight_layout()
        plt.show()
    
    
def testDataLoader(config):
    train_dataset = shapeNetDataset(config["dataset_path"], config["point_cloud_size"], 0, "")
    test_dataset = shapeNetDataset(config["dataset_path"], config["point_cloud_size"], 1, "")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)
    
    #dummy loop
    start = time.perf_counter()

    for pointcloud, pc_class, label, seg_class in train_loader:
        end = time.perf_counter()
        print(f"getting data elapsed time {end-start:6f} seconds")
        showBatchPointcloud(pointcloud, label, pc_class, seg_class)
        start = time.perf_counter()

if __name__ == "__main__": 
    config = {
        #"dataset_path": "/mnt/456c90d8-963b-4daa-a98b-64d03c08e3e1/Black_1TB/datasets/shapenet/PartAnnotation/" ,
        "dataset_path": "F:/AIDL_FP/Datasets/PartAnnotation/",
        "point_cloud_size": 1024,
        "epochs": 1,
        "lr": 1e-3,
        "log_interval": 1000,
        "batch_size": 16
    }

    #showPointCloud("/mnt/456c90d8-963b-4daa-a98b-64d03c08e3e1/Black_1TB/datasets/shapenet/PartAnnotation/")
    testDataLoader(config)