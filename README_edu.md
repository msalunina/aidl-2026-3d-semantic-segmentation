
## PointNet Architecture Validation

The goal of the following experiments is to validate the implementation of the PointNet network [1](https://arxiv.org/pdf/1612.00593)
For this task, we decided to run experiments using ShapeNet dataset as stated in the original work

The first implementation show in [PointNet architecture](###implementation) is used to run a train/test/validation with the ShapeNet dataset


### ShapeNet Part Segmentation Task

The dataset has been downloaded from [Kaggle]( https://www.kaggle.com/datasets/mitkir/shapenet/download?datasetVersionNumber=1) web site as stated in the [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/shapenet.html) documentation.

The dataset is splitted in train,test,validation as shown in the following table.

|        |  Air    |  Bag    |  Cap    |  Car    |  Cha    |  Ear    |  Gui    |  Kni    |  Lam    |  Lap    |  Mot    |  Mug    |  Pis    |  Roc    |  Ska    |  Tab    |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| TRAIN  |  1958.0 (72.8%)  |  54.0  (71.0%)  |  39.0  (70.9%)  |  659.0  (73.4%)  |  2658.0 (70.7%)  |  49.0  (71.0%)  |  550.0  (69.9%)  |  277.0  (70.7%)  |  1118.0  (72.3%)  |  324.0  (71.8%)  |  125.0  (61.9%)  |  130.0  (70.6%)  |  209.0  (73.8%)  |  46.0  (69.7%)  |  106.0  (69.7%)  |  3835.0  (72.8%)  |
|  EVAL  |  341.0  (12.7%) |  14.0  (18.4%) |  11.0  (20.0%) |  158.0  (17.6%) |  704.0  (18.7%) |  14.0  (20.3%) |  159.0  (20.2%) |  80.0  (20.4%) |  286.0  (18.5%) |  83.0  (18.4%) |  51.0  (25.2%) |  38.0  (20.6%) |  44.0  (15.5%) |  12.0  (18.2%) |  31.0  (20.4%) |  848.0  (16.1%) |
|  TEST  |  391.0  (14.5%) |  8.0  (10.5%) |  5.0  (9.1%) |  81.0  (9.0%) |  396.0 (10.5%) |  6.0 (8.7%) |  78.0  (9.9%) |  35.0 (8.9%) |  143.0 (9.2%) |  44.0 (9.8%) | 26.0  (12.9%) |  16.0  (8.7%)   |  30.0 (10.6%)  |  8.0  (12.1%)  |  15.0  (9.9%) |  588.0 (11.1%) |

The goal results for the model are shown in the following table:

|       |  MEAN  |   Air   |   Bag   |   Cap   |   Car   |   Cha   |   Ear   |   Gui   |   Kni   |   Lam   |   Lap   |   Mot   |   Mug   |   Pis   |   Roc   |   Ska   |   Tab   |
|:-----:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| PointNet  |  83.7  |  83.4  | 78.7  | 82.5  |  74.9  |  89.6  |  73.0  |  91.5  |  85.9  |  80.8  |  95.3  |  65.2  |  93.0  |  81.2  |  57.9  |  72.8  |  80.6  |

The measure for the experiments is IoU(%), imlementing a compensated IoU for objects that may have a missing label part. For example plane class has part labels [0,1,2,3] but an object in the batch may have part labels [0,1,2] then a value of 1/4 is added in the IoU metrics, this avoids a punishment in IoU for non present parts in an object. This compensation is mentioned in the original work. 


### Dataloader implementation

To be able to use the compensated IoU measure, and one hote vector class index, we had to create a custom dataloader. This will allow us to return with the element, the compensated IoU value, and the index of the object class that is also needed in the next steps.

The dataloader is based on the [torch_geometrics](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/shapenet.html) implementation, and can be found in [shapenet_dataset.py](https://github.com/msalunina/aidl-2026-3d-semantic-segmentation/blob/main/src/utils/shapenet_dataset.py)


### Experiments

All the experiments will be run using the following configuration:

```yaml
  learning_rate: 0.01          # Initial LR passed to Adam
  scheduler_type: cosine       # Only supported option currently
  scheduler_min_lr: 0.00001    # Floor LR (eta_min in CosineAnnealingLR)
  num_epochs: 50               # Also used as T_max for the scheduler
  num_points: 1024
  batch_size: 32
  random_noise: mean 0 std dev 0.02
  rotation_arround_up_axis: 0.7
```

**1. PointNet base model**

First we do a train/evaluation run 

The base model gives the following results

|       |  MEAN  |   Air   |   Bag   |   Cap   |   Car   |   Cha   |   Ear   |   Gui   |   Kni   |   Lam   |   Lap   |   Mot   |   Mug   |   Pis   |   Roc   |   Ska   |   Tab   |
|:-----:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|TRAIN  |  72.5  |  75.3  |  67.6  |  77.4  |  61.7  |  85.6  |  61.6  |  85.6  |  78.1  |  83.2  |  93.8  |  45.2  |  73.6  |  76.2  |  48.2  |  63.4  |  84.1  |
| EVAL  |  70.3  |  76.4  |  69.0  |  47.1  |  63.4  |  86.9  |  58.5  |  85.9  |  80.3  |  80.2  |  93.6  |  47.0  |  80.9  |  71.1  |  37.6  |  64.2  |  81.9  |

We can see that the results are quite different from the original work. 
Going deep in the original work, they state the following changes in the architecture for improving part segmentation task:

+ Adding a one-hot vector for the object class and concatenate it in the final embedding
+ Adding skip connections and concatenate them in the final embedding
+ increase layer sizes in all the network

The network with the improvements stated in the original work is shown in the following image
![PointNet architecture](figs/part_segmentation_pointnet.png)

With the specified changes, the following experiments are planned:

+ Pointnet + one-hot vector
+ Pointnet + skip connections
+ Pointnet + one-hot vector + skip connections
+ Increase layer sizes on architecture with best results
 

The results for the experiments are shown in the following tables

**2. PointNet + One-Hot vector**

|       |  MEAN  |   Air   |   Bag   |   Cap   |   Car   |   Cha   |   Ear   |   Gui   |   Kni   |   Lam   |   Lap   |   Mot   |   Mug   |   Pis   |   Roc   |   Ska   |   Tab   |
|:-----:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|TRAIN  |  78.1  |  77.9  |  77.9  |  85.1  |  66.4  |  87.0  |  74.9  |  88.2  |  82.9  |  85.8  |  94.3  |  51.7  |  87.0  |  81.2  | 56.4  |  68.4  |  84.9  |
| EVAL  |  76.1  |  78.5  |  69.4  |  74.7  |  68.6  |  88.5  |  71.0  |  87.5  |  83.2  |  83.1  |  94.1  |  54.9  |  89.7  |  77.6  | 43.9  |  70.5  |  83.1  |

**3. PointNet + Skip Connections**

|       |  MEAN  |   Air   |   Bag   |   Cap   |   Car   |   Cha   |   Ear   |   Gui   |   Kni   |   Lam   |   Lap   |   Mot   |   Mug   |   Pis   |   Roc   |   Ska   |   Tab   |
|:-----:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|TRAIN  |  74.9  |  76.3  |  75.7  |  77.9  |  66.5  |  86.2  |  68.6  |  86.6  |  80.5  |  84.5  |  93.9  |  44.2  |  81.0  |  73.8  |  52.5  |  65.0  |  84.6  |
| EVAL  |  72.0  |  77.5  |  71.2  |  51.7  |  68.8  |  87.6  |  65.9  |  86.4  |  81.1  |  81.9  |  93.5  |  43.6  |  84.2  |  71.7  |  41.3  |  63.6  |  82.6  |


**4. PointNet + One-Hot vector + Skip Connections**

|       |  MEAN  |   Air   |   Bag   |   Cap   |   Car   |   Cha   |   Ear   |   Gui   |   Kni   |   Lam   |   Lap   |   Mot   |   Mug   |   Pis   |   Roc   |   Ska   |   Tab   |
|:-----:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|TRAIN  |  80.4  |  79.1  |  76.8  |  89.0  |  72.0  |  87.8  |  76.5  |  88.5  |  83.1  |  86.7  |  94.6  |  57.2  |  87.6  |  82.7  |  61.9  |  78.2  |  85.2  |
| EVAL  |  77.3  |  78.7  |  72.7  |  68.2  |  71.7  |  88.4  |  69.4  |  87.7  |  82.8  |  83.6  |  94.3  |  60.1  |  89.3  |  78.4  |  50.3  |  77.0  |  83.5  |


**5. PointNet + One-Hot vector + Skip Connections + Layer Sizes**

|       |  MEAN  |   Air   |   Bag   |   Cap   |   Car   |   Cha   |   Ear   |   Gui   |   Kni   |   Lam   |   Lap   |   Mot   |   Mug   |   Pis   |   Roc   |   Ska   |   Tab   |
|:-----:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:| 
|TRAIN  |  79.1  |  79.7  |  76.6  |  88.9  |  71.7  |  87.9  |  72.2  |  87.9  |  81.4  |  85.7  |  94.8  |  54.7  |  83.5  |  81.8  |  57.1  |  76.7  |  84.9  |
| EVAL  |  76.4  |  79.1  |  69.0  |  68.5  |  72.0  |  88.7  |  71.2  |  87.4  |  81.8  |  84.3  |  94.5  |  58.0  |  86.9  |  75.9  |  47.0  |  75.2  |  83.4  |


### Conclusions

The best results for part segmentation task in ShapeNet dataset has been achieved using the 3rd configuration, PointNet + One-Hot vector + skip connections, the following table shows the comparison of or best results against the original work, for this comparison we are using the result obtained with the test split

|       |  MEAN  |   Air   |   Bag   |   Cap   |   Car   |   Cha   |   Ear   |   Gui   |   Kni   |   Lam   |   Lap   |   Mot   |   Mug   |   Pis   |   Roc   |   Ska   |   Tab   |
|:-----:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| PointNet  |  83.7  |  83.4  | 78.7  | 82.5  |  74.9  |  89.6  |  73.0  |  91.5  |  85.9  |  80.8  |  95.3  |  65.2  |  93.0  |  81.2  |  57.9  |  72.8  |  80.6  |
| OURS      |  78.2  |  80.9  | 79.3  |  61.1 |  73.4  |  89.1  |  69.6  |  88.7  |  83.4  |  83.0  |  95.8  |  57.5  |  87.9  |  81.5  |  55.1  |  69.5  |  85.6  |


The average IoU(%) achieved is really close to the original work, so we can conclude that our implementation of the PointNet architecture is validated.
