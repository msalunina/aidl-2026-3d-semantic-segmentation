
# POINTNET ARCHITECTURE VALIDATION

The goal of the following experiments is to validate the implementation of the PointNet network [1](https://arxiv.org/pdf/1612.00593)
For this task, we decided to run experiments using ShapeNet dataset as stated in the original work

The first implementation show in [PointNet architecture](###implementation) is used to run a train/test with the ShapeNet dataset


## SETUP

The experiments have been run in a desktop PC:
min 2GB GPU RAM

# SHAPENET PART SEGMENTATION

The dataset has been downloaded from [Kaggle]( https://www.kaggle.com/datasets/mitkir/shapenet/download?datasetVersionNumber=1) web site as stated in the [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/shapenet.html) documentation.

The dataset is splitted in train,test,validation as shown in the following table.

|        |  Air    |  Bag    |  Cap    |  Car    |  Cha    |  Ear    |  Gui    |  Kni    |  Lam    |  Lap    |  Mot    |  Mug    |  Pis    |  Roc    |  Ska    |  Tab    |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| TRAIN  |  1958.0 (72.8%)  |  54.0  (71.0%)  |  39.0  (70.9%)  |  659.0  (73.4%)  |  2658.0 (70.7%)  |  49.0  (71.0%)  |  550.0  (69.9%)  |  277.0  (70.7%)  |  1118.0  (72.3%)  |  324.0  (71.8%)  |  125.0  (61.9%)  |  130.0  (70.6%)  |  209.0  (73.8%)  |  46.0  (69.7%)  |  106.0  (69.7%)  |  3835.0  (72.8%)  |
|  EVAL  |  341.0  (12.7%) |  14.0  (18.4%) |  11.0  (20.0%) |  158.0  (17.6%) |  704.0  (18.7%) |  14.0  (20.3%) |  159.0  (20.2%) |  80.0  (20.4%) |  286.0  (18.5%) |  83.0  (18.4%) |  51.0  (25.2%) |  38.0  (20.6%) |  44.0  (15.5%) |  12.0  (18.2%) |  31.0  (20.4%) |  848.0  (16.1%) |
|  TEST  |  391.0  (14.5%) |  8.0  (10.5%) |  5.0  (9.1%) |  81.0  (9.0%) |  396.0 (10.5%) |  6.0 (8.7%) |  78.0  (9.9%) |  35.0 (8.9%) |  143.0 (9.2%) |  44.0 (9.8%) | 26.0  (12.9%) |  16.0  (8.7%)   |  30.0 (10.6%)  |  8.0  (12.1%)  |  15.0  (9.9%) |  588.0 (11.1%) |

## DATALOADER CONTRBUTIONS

+ compensated IoU for part segmentation
+ class object 

The goal results for the model are shown in the following table:

|       |  MEAN  |   Air   |   Bag   |   Cap   |   Car   |   Cha   |   Ear   |   Gui   |   Kni   |   Lam   |   Lap   |   Mot   |   Mug   |   Pis   |   Roc   |   Ska   |   Tab   |
|:-----:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| PointNet  |  83.7  |  83.4  | 78.7  | 82.5  |  74.9  |  89.6  |  73.0  |  91.5  |  85.9  |  80.8  |  95.3  |  65.2  |  93.0  |  81.2  |  57.9  |  72.8  |  80.6  |


The base model gives the following results

|       |  MEAN  |   Air   |   Bag   |   Cap   |   Car   |   Cha   |   Ear   |   Gui   |   Kni   |   Lam   |   Lap   |   Mot   |   Mug   |   Pis   |   Roc   |   Ska   |   Tab   |
|:-----:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|TRAIN  |  0.725  |  0.753  |  0.676  |  0.774  |  0.617  |  0.856  |  0.616  |  0.856  |  0.781  |  0.832  |  0.938  |  0.452  |  0.736  |  0.762  |  0.482  |  0.634  |  0.841  |
| EVAL  |  0.703  |  0.764  |  0.690  |  0.471  |  0.634  |  0.869  |  0.585  |  0.859  |  0.803  |  0.802  |  0.936  |  0.470  |  0.809  |  0.711  |  0.376  |  0.642  |  0.819  |

We can see that the results are quite different from the original work. 
Going deep in the original work, they state the following changes in the architecture for improving part segmentation:
+ Adding a one-hot vector for the object class
+ compensation in the IoU for the parts that are missing in a specific object,
+ concatenating results from the first convolutional layers
+ increase layer sizes in all the network

The network with the improvements stated in the original work is shown in the following image
![PointNet architecture](figs/part_segmentation_pointnet.png)

With the specified changes, the following experiments are planned:

+ Pointnet + one-hot vector
+ Pointnet + skip connections
+ Pointnet + one-hot vector + skip connections
+ Increase layer sizes on architecture with best results

The results for the experiments are shown in the following tables

## POINET + ONE HOT VECTOR 

|       |  MEAN  |   Air   |   Bag   |   Cap   |   Car   |   Cha   |   Ear   |   Gui   |   Kni   |   Lam   |   Lap   |   Mot   |   Mug   |   Pis   |   Roc   |   Ska   |   Tab   |
|:-----:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|TRAIN  |  0.78  |  0.779  |  0.779  |  0.851  |  0.664  |  0.870  |  0.749  |  0.882  |  0.829  |  0.858  |  0.943  |  0.517  |  0.870  |  0.812  |  0.564  |  0.684  |  0.849  |
| EVAL  |  0.76  |  0.785  |  0.694  |  0.747  |  0.686  |  0.885  |  0.710  |  0.875  |  0.832  |  0.831  |  0.941  |  0.549  |  0.897  |  0.776  |  0.439  |  0.705  |  0.831  |

## POINTNET + SKIP CONN 

|       |  MEAN  |   Air   |   Bag   |   Cap   |   Car   |   Cha   |   Ear   |   Gui   |   Kni   |   Lam   |   Lap   |   Mot   |   Mug   |   Pis   |   Roc   |   Ska   |   Tab   |
|:-----:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|TRAIN  |  0.75  |  0.763  |  0.757  |  0.779  |  0.665  |  0.862  |  0.686  |  0.866  |  0.805  |  0.845  |  0.939  |  0.442  |  0.810  |  0.738  |  0.525  |  0.650  |  0.846  |
| EVAL  |  0.72  |  0.775  |  0.712  |  0.517  |  0.688  |  0.876  |  0.659  |  0.864  |  0.811  |  0.819  |  0.935  |  0.436  |  0.842  |  0.717  |  0.413  |  0.636  |  0.826  |


## POINTNET + OHV + SKIP CONN 

|       |  MEAN  |   Air   |   Bag   |   Cap   |   Car   |   Cha   |   Ear   |   Gui   |   Kni   |   Lam   |   Lap   |   Mot   |   Mug   |   Pis   |   Roc   |   Ska   |   Tab   |
|:-----:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|TRAIN  |  0.80  |  0.791  |  0.768  |  0.890  |  0.720  |  0.878  |  0.765  |  0.885  |  0.831  |  0.867  |  0.946  |  0.572  |  0.876  |  0.827  |  0.619  |  0.782  |  0.852  |
| EVAL  |  0.77  |  0.787  |  0.727  |  0.682  |  0.717  |  0.884  |  0.694  |  0.877  |  0.828  |  0.836  |  0.943  |  0.601  |  0.893  |  0.784  |  0.503  |  0.770  |  0.835  |
| TEST  |  0.78  |  0.809  |  0.793  |  0.611  |  0.734  |  0.891  |  0.696  |  0.887  |  0.834  |  0.830  |  0.958  |  0.575  |  0.879  |  0.815  |  0.551  |  0.695  |  0.856  |


## POINTNET - PAPER - big layer sizes - ohv - skip links
|       |  MEAN  |   Air   |   Bag   |   Cap   |   Car   |   Cha   |   Ear   |   Gui   |   Kni   |   Lam   |   Lap   |   Mot   |   Mug   |   Pis   |   Roc   |   Ska   |   Tab   |
|:-----:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:| 
|TRAIN  |  0.79  |  0.797  |  0.766  |  0.889  |  0.717  |  0.879  |  0.722  |  0.879  |  0.814  |  0.857  |  0.948  |  0.547  |  0.835  |  0.818  |  0.571  |  0.767  |  0.849  |
| EVAL  |  0.76  |  0.791  |  0.690  |  0.685  |  0.720  |  0.887  |  0.712  |  0.874  |  0.818  |  0.843  |  0.945  |  0.580  |  0.869  |  0.759  |  0.470  |  0.752  |  0.834  |


## POINTNET + ONE HOT VECTOR + SKIP CONNECTIONS + weights

|       |  MEAN  |   Air   |   Bag   |   Cap   |   Car   |   Cha   |   Ear   |   Gui   |   Kni   |   Lam   |   Lap   |   Mot   |   Mug   |   Pis   |   Roc   |   Ska   |   Tab   |
|:-----:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|TRAIN  |  0.73  |  0.760  |  0.656  |  0.809  |  0.581  |  0.833  |  0.712  |  0.850  |  0.809  |  0.831  |  0.943  |  0.449  |  0.695  |  0.742  |  0.500  |  0.652  |  0.822  |
| EVAL  |  0.71  |  0.752  |  0.662  |  0.734  |  0.593  |  0.853  |  0.650  |  0.842  |  0.813  |  0.813  |  0.935  |  0.448  |  0.732  |  0.732  |  0.369  |  0.651  |  0.785  |
| TEST  |  0.71  |  0.780  |  0.709  |  0.702  |  0.589  |  0.856  |  0.637  |  0.857  |  0.809  |  0.811  |  0.948  |  0.445  |  0.671  |  0.717  |  0.483  |  0.576  |  0.827  |

## CONCLUSIONS

ShapeNet dataset for part segmentation does not need weights eventought some classes have high imbalance

|       |  Air   |   Bag   |   Cap   |   Car   |   Cha   |   Ear   |   Gui   |   Kni   |   Lam   |   Lap   |   Mot   |   Mug   |   Pis   |   Roc   |   Ska   |   Tab   |
|:-----:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| IMBALANCE | 4.940 |  14.219 | 2.824 |  13.313 |  11.485| 4.780 | 8.589|    1.0305|  49.615|     1.153 |  100.970 |    15.392 | 11.791|  6.098 |  11.615|   24.364| 