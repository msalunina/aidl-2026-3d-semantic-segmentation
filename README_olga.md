
# METRICS

Evaluating the performance of a semantic segmentation model requires metrics that measure how well the predicted labels match the ground truth labels for each element in the input. In the case of point cloud segmentation, the task consists of assigning a semantic class to every point in the cloud. While classification accuracy measures the proportion of correctly labeled points, it is often not a reliable metric for segmentation tasks because datasets are usually highly imbalanced. For example, in LiDAR datasets large portions of the scene may correspond to dominant classes such as ground or vegetation, while other classes such as vehicles or utilities appear much less frequently.

In such situations, a model could achieve high accuracy simply by predicting the dominant classes, even if it fails to correctly segment rare classes. For this reason, Intersection over Union (IoU) is widely used as a more robust metric for segmentation evaluation.

## Intersection over Union (IoU)
Intersection over Union ($IoU$) measures the overlap between the predicted region for a class and the corresponding ground truth region. The $IoU$ for a class is defined as

$$
IoU = \frac{TP}{TP + FP + FN}
$$

where:

- **TP (True Positives):** points that belong to the given class and are correctly predicted by the model ("correct predictions")

- **FP (False Positives):** points that the model predicts as belonging to the given class, but whose true label is different ("incorrect predictions")

- **FN (False Negatives):** points that belong to the given class, but are incorrectly predicted by the model ("missed detections")


Example: _IoU for building_
- TP: a point labeled as _building_ that is also predicted as _building_
- FP: a point labeled as _vegetation_ that is predicted as _building_
- FN: a point labeled as _building_ that is predicted as _vegetation_

This formula can also be interpreted as the ratio between the **intersection** and the **union** of the predicted and ground truth sets of points:

- the **intersection** corresponds to the points correctly predicted as belonging to the class (TP)

- the **union** corresponds to all points that belong to the class either in the prediction or in the ground truth (TP + FP + FN)



The $IoU$ value ranges from 0 to 1, where:

- 1 indicates perfect overlap between prediction and ground truth
- 0 indicates no overlap

## Mean Intersection over Union (mIoU)

In semantic segmentation tasks, $IoU$ is computed independently for each class. To obtain an overall measure of segmentation performance across all classes, the mean Intersection over Union ($mIoU$) is used.
The $mIoU$ is defined as the average $IoU$ over all classes:
$$
mIoU = \frac{1}{C} \sum_{c=1}^{C} IoU_c
$$

where:
- $C$ is the number of classes
- $IoU_c$ is the IoU for class $c$

By averaging over classes, mIoU ensures that all classes contribute equally to the evaluation, preventing dominant classes from disproportionately influencing the metric. This makes mIoU particularly suitable for segmentation tasks in datasets with class imbalance, such as LiDAR point cloud datasets.

Unlike accuracy, which can be dominated by frequent classes, mIoU evaluates segmentation performance independently for each class and therefore provides a more reliable measure of overall segmentation quality.



## IoU Implementation

Intersection and union for each class are accumulated for all batches across the entire epoch and one final value is obtained per epoch and class:

$$
IoU_c = \frac{\sum intersection_c}{\sum union_c}
$$

where the $\sum$ represents all batches. Then, the average of all $IoU_c$ is computed to obtain $mIoU$



# POINTNET++

PointNet++ is a deep neural network designed to process unordered point sets sampled from a metric space. The architecture extends the original PointNet by introducing 
a hierarchical feature learning framework that captures both local geometric structures and global contextual information.

While PointNet processes the entire point cloud using a single global aggregation, PointNet++ organizes the computation into multiple levels of abstraction, progressively 
learning features from small local neighborhoods to larger spatial regions. This hierarchical structure enables the network to capture fine-grained geometric patterns and 
improves performance on complex scenes and segmentation tasks.

The architecture consists of two main components:

- Encoder (Set Abstraction layers, SA) – extracts hierarchical features from the input point cloud.
- Decoder (Feature Propagation layers, FP) – propagates the learned features back to the original points for dense prediction tasks such as semantic segmentation.

## ENCODER :
The encoder builds a hierarchical representation of the point cloud through successive **Set Abstraction (SA)** layers. Each SA layer applies sequentially:

1. **Farthest Point Sampling (FPS)** to select a subset of centers  
2. **Neighborhood grouping** around each center  
3. A **shared MLP + max pooling** to extract local geometric features  (mini PointNet)

As the network progresses through the hierarchy (towards deeper layers), the number of centers decreases and the distance between them increases.  
Therefore, even if the number of neighbors remains constant, the spatial extent of each neighborhood (effective receptive field) naturally grows with depth, allowing each 
layer to capture geometric structures at a different spatial scale. Early layers focus on **small local structures (fine geometric structures)**, while deeper layers represent **larger semantic structures** describing the scene.

For instance, for our DALES dataset:

- **SA1** may capture fine geometric patterns such as edges, corners, or small object parts (e.g., parts of a car wheel or tree branches).
- **SA2** may capture slightly larger structures, such as parts of objects (e.g., a full car roof or part of a tree canopy).
- **SA3** may capture complete objects or object groups (e.g., an entire car or a section of vegetation).
- **SA4** may represent larger scene elements, such as a **car within a street segment**, a **group of trees**, or a **portion of terrain**.

### 1. Farthest Point Sampling (FPS)

Explain Farthest Point Sampling (FPS)


### 2. Neighborhood grouping

The grouping stage constructs local neighbourhoods around each center. The strategy used to select neighbors determines the **spatial support of the local patch**,
which directly influences the type of geometric structures the network can capture.

Two grouping strategies are compared:

- **k-Nearest Neighbors**: selects the K closest points
- **Radius-Based grouping**: selects all points within a predefined radius

---

#### k-Nearest Neighbors (`knn`)

In knn grouping, for each center point the K nearest points in Euclidean space are selected.
Consequently:
- **Number of neighbors** is fixed
- **Spatial size of the neighborhood** depends on the local point density. 

In dense regions, the K nearest neighbors lie close to the center and defie a small spatial patch. Instead, in sparse regions, the same number of neighbors may lie 
further defining a much larger spatial area. As a result, the **effective receptive field** varies with point density.

As a result, although the network always processes the same number of points, the geometric scale represented by those points changes across the scene, 
making the kNN grouping density-dependent in terms of spatial coverage.

In deeper SA layers, FPS progressively reduces the number of centers, increasing the spacing between them. 
Consequently, the spatial extent of the neighborhoods naturally grows with depth, even when the value of K remains constant. 
This allows deeper layers to capture larger geometric structures, such as buildings or terrain.



#### Radius-Based Grouping

The idea behind Radius-Based grouping is to reduce the density dependency.

In Radius-Based grouping, neighbors are selected within a **fixed spatial radius** around each center. 
Consequently:
- **Maximum spatial support of the neigborhood** is fixed for each layer
- **Number of neighbors** depends on the local point density

However, since PointNet++ expects a fixed number of neighbors, only up to K neighbors are retained. 
Therefore, the **effective receptive field** may still depend on how those neighbors are selected, especially in dense regions where more than K points may lie inside the ball.
Two strategies were implemented to select these K neighbors.

- `ball_closest`: 

    When more than K points lie inside the radius, only the **K closest points to the center** are selected.

    Although the neighborhood is bounded by the radius, this selection strategy introduces a bias toward points located near the center. 
    As a consequence, the **effective receptive field** may shrink in dense regions introducing a **mild density dependence**

- `ball_random`: 

    When more than K points lie inside the radius, **K of them are randomly sampled**.

    In this case, the radius effctively defines the spatial extent of the neighborhood, since any point within the ball has an equal probability 
    of being selected. As a consequence, the **effective receptive field** closely matches the radius making the strategy largely **density independent**

If the ball contains less than K points, some points are repeated. Note that it does not bias information towards repeated points since we deliberately apply 
max-pooling to extract features (instead of average-pooling).   

The following comparison highlights how the choice of grouping strategy controls the spatial scale of the features learned by PointNet++, 
and therefore influences the types of geometric structures that can be captured at different levels of the network hierarchy.


| Method        | Spatial Support | Effective Receptive Field        | Density Sensitivity |
|---------------|-----------------|-----------------------------------|---------------------|
| **knn**       | Variable (no limit)      | Variable                          | High                |
| **ball_closest** |Fixed (radius cap)      | May shrink in dense regions (center-biased sampling)      | Moderate            |
| **ball_random**  | Fixed (radius cap)    | Tends to span the ball (random sampling)     | Lower               |


### 3. A **shared MLP + max pooling**




## Configurable options:

### SetAbstraction (SA) layer
Number of neighbours per center for each SA layer:
- K = [32,32,32,32] (default)
Radius size of each SA layer:
- radius = [0.08,0.1,0.2,0.4] (default)
grouping options:
- 'knn': for each center, choose the K closest neighbours
- 'ball_closest': for each center, find up to K neighbours within a ball of radius 'radius'. If inside > K, chooses the K closest. If inside < K, repeat some up to K
- 'ball_random': for each center, find up to K neighbours within a ball of radius 'radius'. If inside > K, randomly sample among them. If inside < K, repeat some up to K


## EXPERIMENTS

Table 1. Summary of the PointNet++ experiment configurations. Each experiment modifies a specific component of the baseline model: the dropout rate, the number of neighbors, the grouping strategy or the input feature channels

| PointNet++|      test     |    grouping    |    dropout   |      K-neighbors      |              feature channels               | 
|:---------:|:-------------:|:--------------:|:------------:|:-----------------------:|:------------------------------------:|
|     1     |  baseline     |      "knn"     |      0.5     | [32,32,32,32] (exact K) | [xyz,return_number,number_of_returns]|
|     2     |    dropout    |      "knn"     |      0.3     | [32,32,32,32] (exact K) | [xyz,return_number,number_of_returns]|
|     3     |  K-neighbors  |      "knn"     |      0.5     | [32,32,64,64] (exact K) | [xyz,return_number,number_of_returns]| 
|     4     |  grouping     | "ball_closest" |      0.5     | [32,32,32,32] (max K)   | [xyz,return_number,number_of_returns]| 
|     5     |  grouping     | "ball_random"  |      0.5     | [32,32,32,32] (max K)   | [xyz,return_number,number_of_returns]| 
|     6     |  channels     |      "knn"     |      0.5     | [32,32,32,32] (exact K) | [xyz]                                | 


Note: for the two ball-based grouping strategies, the parameter K does not define the exact number of neighbors but instead, specifies the maximum number of neighbours 
that can be selected within the ball. The effective size of the neighborhood is controlled by the radius parameter, which is fixed in all experiments to [0.08, 0.1, 0.2, 0.4]. 
These values were selected based on preliminary exploratory tests.


### Focal loss + class weighting 

| Metric | 1. baseline<br>olga | 1. baseline<br>edu | 1. baseline<br>dygiro |
|------|------|------|------|
| **Overall metrics** |||||||
| mIoU | 0.781 / 0.766 | 0.783 / 0.768 | 0.781 / **0.771** |
| Loss | 0.019 / **0.019** | 0.019 / **0.019** | 0.019 / 0.020 |
| Accuracy | 0.953 / 0.951 | 0.954 / 0.951 | 0.953 / 0.951 |
| **Class IoU** |||||||
| Ground | 0.947 / **0.942** | 0.947 / **0.942** | 0.947 / **0.942** |
| Vegetation | 0.853 / 0.849 | 0.854 / 0.849 | 0.854 / **0.851** |
| Buildings | 0.950 / **0.947** | 0.950 / **0.947** | 0.950 / 0.944 |
| Vehicle | 0.619 / 0.567 | 0.621 / 0.551 | 0.615 / **0.569** |
| Utility | 0.537 / 0.526 | 0.543 / **0.551** | 0.538 / 0.547 |

**Table 1**. Comparing baseline on different machines. Focal loss + class weighting [0.2553, 0.3465, 0.4482, 1.8602, 2.0897].<br>
All experiments use class-aware sampler. Values are reported as train / validation. Bold values indicate the best validation score.


| Metric | 1 - baseline<br>olga | 2 - dropout<br>edu | 3 - K-neighbors<br>olga | 4 - ball_closest<br>edu | 5 - ball_random<br>olga | 6 - xyz only<br>edu |
|------|------|------|------|------|------|------|
| **Overall metrics** |||||||
| mIoU | 0.781 / 0.766 | 0.785 / 0.768 | 0.782 / 0.769 | 0.776 / **0.771** | 0.772 / 0.766 | 0.767 / 0.754 |
| Loss | 0.019 / 0.019 | 0.018 / 0.021 | 0.019 / 0.019 | 0.020 / **0.018** | 0.021 / 0.019 | 0.021 / 0.020 |
| Accuracy | 0.953 / 0.951 | 0.954 / 0.950 | 0.954 / **0.952** | 0.952 / **0.952** | 0.951 / 0.951 | 0.949 / 0.948 |
| **Class IoU** |||||||
| Ground | 0.947 / 0.942 | 0.947 / 0.940 | 0.947 / **0.943** | 0.946 / **0.943** | 0.944 / 0.942 | 0.941 / 0.938 |
| Vegetation | 0.853 / 0.849 | 0.856 / 0.845 | 0.854 / 0.851 | 0.851 / **0.855** | 0.849 / 0.852 | 0.844 / 0.843 |
| Buildings | 0.950 / 0.947 | 0.950 / 0.942 | 0.950 / **0.948** | 0.943 / **0.948** | 0.943 / 0.947 | 0.945 / 0.947 |
| Vehicle | 0.619 / 0.567 | 0.626 / 0.570 | 0.622 / 0.560 | 0.610 / **0.595** | 0.599 / 0.584 | 0.591 / 0.543 |
| Utility | 0.537 / 0.526 | 0.544 / 0.541 | 0.538 / 0.541 | 0.531 / 0.515 | 0.525 / 0.503 | 0.511 / 0.501 |

**Table 2**. Comparison of PointNet++ configurations: Focal loss + class weighting [0.2553, 0.3465, 0.4482, 1.8602, 2.0897]. 
<br>All experiments use class-aware sampler. Values are reported as train / validation. Bold values indicate the best validation score.

### NLL loss + near-uniform class weighting 

| Metric | 1 - baseline | 2 - dropout | 3 - K-neighbors | 4 - ball_closest | 5b - ball_random | 6 - xyz only |
|------|------|------|------|------|------|------|
| **Overall metrics** |||||||
| mIoU | 0.813 / 0.796 | 0.816 / 0.804 | 0.813 / 0.803 | 0.806 / **0.811** | 0.804 / 0.805 | 0.795 / 0.796 |
| Loss | 0.116 / 0.122 | 0.114 / 0.120 | 0.115 / 0.122 | 0.120 / **0.116** | 0.122 / 0.117 | 0.126 / 0.124 |
| Accuracy | 0.959 / 0.956 | 0.959 / 0.956 | 0.959 / 0.956 | 0.957 / **0.957** | 0.957 / **0.957** | 0.955 / 0.954 |
| **Class IoU** |||||||
| Ground | 0.950 / 0.944 | 0.950 / 0.944 | 0.950 / 0.943 | 0.949 / **0.945** | 0.948 / **0.945** | 0.945 / 0.939 |
| Vegetation | 0.868 / 0.861 | 0.869 / 0.862 | 0.868 / 0.860 | 0.864 / **0.867** | 0.863 / 0.866 | 0.857 / 0.858 |
| Buildings | 0.952 / 0.949 | 0.953 / 0.949 | 0.952 / 0.948 | 0.946 / 0.950 | 0.946 / **0.952** | 0.948 / 0.948 |
| Vehicle | 0.680 / 0.640 | 0.688 / 0.654 | 0.682 / 0.654 | 0.672 / **0.675** | 0.666 / 0.665 | 0.646 / 0.646 |
| Utility | 0.615 / 0.586 | 0.618 / 0.611 | 0.612 / 0.611 | 0.600 / **0.617** | 0.596 / 0.599 | 0.582 / 0.592 |


**Table 3**. Comparison of PointNet++ configurations: NLL loss + near-uniform class weighting [0.9894, 0.9894, 0.9894, 1.0049, 1.0270]
<br>All experiments use class-aware sampler. Values are reported as train / validation. Bold values indicate the best validation score.



### FINAL CONFIG (optimal for PoinNet)

| Metric | 1 - baseline | 2 - dropout | 3 - K-neighbors | 4 - ball_closest | 5 - ball_random | 6 - xyz only | 7 - experiment |
|------|------|------|------|------|------|------|------|
| **Overall metrics** ||||||||
| mIoU | — | — | — | — | — | — | 0.806 / 0.799 |
| Loss | — | — | — | — | — | — | 0.131 / 0.130 |
| Accuracy | — | — | — | — | — | — | 0.957 / 0.956 |
| **Class IoU** ||||||||
| Ground | — | — | — | — | — | — | 0.949 / 0.945 |
| Vegetation | — | — | — | — | — | — | 0.864 / 0.863 |
| Buildings | — | — | — | — | — | — | 0.948 / 0.950 |
| Vehicle | — | — | — | — | — | — | 0.675 / 0.657 |
| Utility | — | — | — | — | — | — | 0.594 / 0.578 |





"knn" vs "ball_closest"/"ball_random" is not a fair comparison because:
With kNN:
- you always get exactly K points
- the spatial radius is variable (no limit)
- neighborhood effective size expands or shrinks depending on density

With ball query:
- you fix the maximum spatial radius 
- the number of points is variable (up to max K)
- the neighborhood effective size 
    - closest: biased toward center when many points exist (may shrink)
    - random: tends to sample the full ball more uniformly

knn is robust in sparse areas (always enough points), but spatial neighborhood varies a lot
ball has a consistent geometric scale (bounding), but may contain very few points.
If very dense areas:
ball_closest → biased toward center 
ball_random  → more spatially uniform sampling


### RESULTS:
