
# METRICS

Evaluating the performance of a semantic segmentation model requires metrics that measure how well the predicted labels match the ground truth labels. In the case of point cloud segmentation, the task consists of assigning a semantic class to every point in the cloud. While classification accuracy measures the proportion of correctly labeled points, it is often not a reliable metric for segmentation tasks because datasets are usually highly imbalanced. For example, in LiDAR datasets large portions of the scene may correspond to dominant classes such as ground or vegetation, while other classes such as vehicles or utilities appear much less frequently.

In such situations, a model could achieve high accuracy simply by predicting the dominant classes, even if it fails to correctly predict rare classes. For this reason, Intersection over Union (IoU) is widely used as a more robust metric for segmentation evaluation.

## Intersection over Union (IoU)
Intersection over Union (IoU) measures the overlap between the predicted region for a class and the corresponding ground truth region. The IoU for a class is defined as

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


## Mean Intersection over Union (mIoU)

In semantic segmentation tasks, IoU is computed independently for each class. To obtain an overall measure of segmentation performance across all classes, the mean Intersection over Union (mIoU) is used.
The mIoU is defined as the average IoU over all classes:

$$
mIoU = \frac{1}{C} \sum_{c=1}^{C} IoU_c
$$

where:
- $C$ is the number of classes
- $IoU_c$ is the IoU for class $c$

By averaging over classes, mIoU ensures that all classes contribute equally to the evaluation, preventing dominant classes from disproportionately influencing the metric. This makes mIoU particularly suitable for segmentation tasks in datasets with class imbalance. Unlike accuracy, which can be dominated by frequent classes, mIoU evaluates segmentation performance independently for each class and therefore provides a more reliable measure of overall segmentation quality.



## IoU Implementation


For each class, the intersection and the union are computed for every batch and accumulated over the entire epoch. Then, their ratio gives a single IoU value per class and epoch:

$$
IoU_c = \frac{\sum_{b} intersection_c}{\sum_{b} union_c}
$$

where the sum is perfomed over all batches b in the epoch. 



# POINTNET++

PointNet++ is a deep neural network designed to process unordered point sets sampled from a metric space. The architecture extends the original PointNet by introducing 
a hierarchical feature learning framework that captures both local geometric structures and global contextual information.

While PointNet processes the entire point cloud using a single global aggregation, PointNet++ organizes the computation into multiple levels of abstraction, progressively 
learning features from small local neighborhoods to larger spatial regions. This hierarchical structure enables the network to capture fine-grained geometric patterns and 
improves performance on complex scenes and segmentation tasks.

The architecture consists of two main components:

- Encoder (Set Abstraction layers, SA) – extracts hierarchical features from the input point cloud.
- Decoder (Feature Propagation layers, FP) – propagates the learned features back to the original points for dense prediction tasks such as semantic segmentation.

Next Figure shows the PointNet++ architecture 

![PointNet++ architecture](figs/pnpp_architecture.png)
(from "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space" (Qi et al., 2017).)




## ENCODER
The encoder builds a hierarchical representation of the point cloud through successive **Set Abstraction (SA)** layers. Each SA layer applies sequentially:

1. **Farthest Point Sampling (FPS)** to select a subset of centers  
2. **Neighborhood grouping** around each center  
3. A **shared MLP + max pooling** to extract local geometric features  (mini PointNet)

As the network progresses through the hierarchy (towards deeper layers), the number of centers decreases and the distance between them increases. Therefore, even if the number of neighbors remains constant, the spatial extent of each neighborhood (effective receptive field) naturally grows with depth, allowing each layer to capture geometric structures at a different spatial scale. Early layers focus on **small local structures (fine geometric structures)**, while deeper layers represent **larger semantic structures** describing the scene.

For instance, for our DALES dataset:

- SA1 may capture fine geometric patterns such as edges, corners, or small object parts
- SA2 may capture slightly larger structures, such as parts of objects 
- SA3 may capture complete objects or object groups 
- SA4 may represent larger scene elements, such as a car within a street segment

### 1. Farthest Point Sampling (FPS)

In order to select the center points, PointNet++ uses Farthest Point Sampling (FPS). Iteratively it selects points that maximize the distance from previously selected centers. This ensures that the sampled points are evenly distributed across the point cloud.


### 2. Neighborhood grouping

The grouping stage constructs local neighbourhoods around each center. The strategy used to select neighbors determines the **spatial support of the local patch**, which directly influences the type of geometric structures the network can capture. Two grouping strategies are compared:

- **k-Nearest Neighbors**: selects the K closest points
- **Radius-Based grouping**: selects all points within a predefined radius


#### k-Nearest Neighbors (`knn`)

In knn grouping, for each center point the K nearest points in Euclidean space are selected. Consequently:
- **Number of neighbors** is fixed
- **Spatial size of the neighborhood** depends on the local point density. 

In dense regions, the K nearest neighbors lie close to the center and defie a small spatial patch. Instead, in sparse regions, the same number of neighbors may lie further defining a much larger spatial area. As a result, the **effective receptive field** varies with point density.

#### Radius-Based Grouping

The idea behind Radius-Based grouping is to reduce the density dependency. To do so, neighbors are selected within a **fixed spatial radius** around each center. Consequently:
- **Maximum spatial support of the neigborhood** is fixed for each layer
- **Number of neighbors** depends on the local point density

However, PointNet++ expects a fixed number of neighbours. If the ball contains less than K points, some points are repeated (note that this repetition does not bias information towards repeated points since in FP layers we deliberately apply max-pooling to extract features instead of average-pooling). On the contrary, if the ball contains more than K points, only up to K are retained. As a consequence, the **effective receptive field** may still depend on how those neighbors are selected, especially in dense regions where more than K points may lie inside the ball. Two strategies were implemented to select these K neighbors.

- `ball_closest`: When more than K points lie inside the radius, only the **K closest points to the center** are selected.

    Although the neighborhood is bounded by the radius, this selection strategy introduces a bias toward points located near the center. 
    As a consequence, the **effective receptive field** may shrink in dense regions introducing a **mild density dependence**

- `ball_random`: When more than K points lie inside the radius, **K of them are randomly sampled**.

    In this case, the radius effctively defines the spatial extent of the neighborhood, since any point within the ball has an equal probability of being selected. As a consequence, the **effective receptive field** closely matches the radius making the strategy largely **density independent**


The following comparison highlights how the choice of grouping strategy controls the spatial scale of the features learned by PointNet++, and therefore influences the types of geometric structures that can be captured at different levels of the network hierarchy.


| Method        | Spatial Support | Effective Receptive Field        | Density Sensitivity |
|---------------|-----------------|-----------------------------------|---------------------|
| **knn**       | Variable (no limit)      | Variable                          | High                |
| **ball_closest** |Fixed (radius cap)      | May shrink in dense regions (center-biased sampling)      | Moderate            |
| **ball_random**  | Fixed (radius cap)    | Tends to span the ball (random sampling)     | Lower               |



### 3. A **shared MLP + max pooling**

Each local region is processed by a mini-PointNet network, which learns a feature representation for the neighborhood. This consists of a shared multilayer perceptron (MLP) applied independently to each point followed by a symmetric aggregation function (max pooling) to obtain a single feature vector representing the region.





## DECODER

While the encoder progressively reduces the number of points and extracts higher-level features, semantic segmentation requires a prediction for **every original input point**. Therefore, PointNet++ includes a decoder composed of successive **Feature Propagation (FP)** layers, which progressively upsample features from sparse point sets back to denser ones.

Each Feature Propagation layer applies sequentially:

1. **3-NN interpolation** to transfer features from a sparse set of points to a denser one  
2. **Concatenation with skip features** coming from the encoder  
3. A **shared MLP** to refine the propagated features  

In this way, the decoder combines high-level semantic information from deeper layers with fine geometric details preserved by the early encoder layers.


### 1. 3-NN Interpolation

At each decoder stage, features from a sparse set of points are interpolated onto a denser set of target points. For each target point, the three nearest source points are identified and their features are interpolated using weights inversely proportional to their distance. Therefore, closer source points contribute more strongly to the interpolated feature, while farther ones contribute less.

This interpolation step allows features learned at coarse spatial resolutions to be transferred back to denser point sets.

### 2. Concatenation with Skip Features

After interpolation, the propagated features are concatenated with the corresponding **skip features** coming from the encoder.

These skip features provide local geometric information extracted at earlier abstraction levels, where the point resolution is still relatively high. Consequently, the decoder does not rely only on coarse semantic information from deep layers, but also reuses fine-grained spatial details that may have been lost during downsampling.

### 3. Shared MLP Refinement

Once the interpolated features and skip features have been concatenated, the resulting feature vectors are refined using a **shared multilayer perceptron (MLP)** applied independently to each point. In doing so, semantic information coming from the decoder and geometric details preserved from the skip connections are fused together, producing a more informative point-wise representation

By stacking several Feature Propagation layers, the decoder progressively reconstructs point features at increasing resolutions until features are available for the full original point cloud size.



## CONFIGURATION


### Encoder configuration (SA Layers)

| Layer | # Centers (FPS) | Neighborhood | K | Radius | MLP |
|:------:|:-----------------:|:-------------:|:---:|:-------:|:------|
| **SA1** | 1024 | knn / ball query | 32 | 0.08 | `[32,32,64]` |
| **SA2** | 256 | knn / ball query | 32 | 0.10 | `[64,64,128]` |
| **SA3** | 64 | knn / ball query | 32 | 0.20 | `[128,128,256]` |
| **SA4** | 16 | knn / ball query | 32 | 0.40 | `[256,256,512]` |

#### Tensor Shapes

Each Set Abstraction (SA) layer samples centers using FPS, groups K neighbors around each center, applies a shared MLP to the grouped features, and aggregates the neighborhood using max pooling.

| Layer | Input<br>(xyz / features)  |  Grouped neighbors | After shared MLP | After max pooling | Output<br>(xyz / features) |
|:------:|:--------------------------|:------------------|:-----------------|:------------------|:------------------------|
| **SA** (generic) | `[B,N,3]`/`[B,N,C]`  | `[B,S,K,3+C]`      |    `[B,C_out,S,K]`  |     `[B,C_out,S]`    | `[B,S,3]`/`[B,S,C_out]` |                       
| **SA1** | `[B,N,3]`/<br>`None`           | `[B,1024,32,3]` | `[B,64,1024,32]` | `[B,64,1024]` | `[B,1024,3]`/<br>`[B,1024,64]` |
| **SA2** | `[B,1024,3]`/<br>`[B,1024,64]` | `[B,256,32,67]` | `[B,128,256,32]` | `[B,128,256]` | `[B,256,3]`/<br>`[B,256,128]` |
| **SA3** | `[B,256,3]`/<br>`[B,256,128]`  | `[B,64,32,131]` | `[B,256,64,32]` | `[B,256,64]` | `[B,64,3]`/<br>`[B,64,256]` |
| **SA4** | `[B,64,3]`/<br>`[B,64,256]`    | `[B,16,32,259]` | `[B,512,16,32]` | `[B,512,16]` | `[B,16,3]`/<br>`[B,16,512]` |

where:
- B: batch size
- N: number of input points 
- S: number of centers
- K: number of neighbors
- C: features/channels


### Decoder configuration (FP layers)

| Layer | Interpolation | Skip connection | MLP |
|:------:|:---------------:|:----------------:|:------|
| **FP4** | 3-NN (16 → 64) | SA3 features | `[256,256]` |
| **FP3** | 3-NN (64 → 256) | SA2 features | `[256,256]` |
| **FP2** | 3-NN (256 → 1024) | SA1 features | `[256,128]` |
| **FP1** | 3-NN (1024 → N) | input features (if any) | `[128,128,128]` |
| **Classifier head** | – | – | `[128,128,num_classes]` |

_NOTE: In the original PointNet++ semantic segmentation architecture, the final decoder FP layer is [128,128,128,128,num_classes].
In our implementation, instead, this last FP is split between the **FP1** block and a separate **classifier head** to allow dropout to be applied explicitly to the two final layers before the per-point class score prediction
As a consequence, the implemented architecture contains one additional 128-dimensional transformation, resulting in [128,128,128,128,128,num_classes]._


#### Tensor Shapes

Each Feature Propagation (FP) layer interpolates features from a sparse point set to a denser one using 3-NN interpolation with inverse-distance weighting, concatenates the interpolated features with skip features from the encoder, and refines them using a shared MLP.


| Layer | Source points<br>(features) | Target points<br>(skip features) | Interpolated source<br>features at target points | After concat (skip) | After shared MLP |
|:------:|:----------------|:-----------------------|:---------------------|:-------------|:----------------|
| **FP** (generic) | `[B,N_s,C_s]` | `[B,N_t,C_skip]` | `[B,N_t,C_s]` | `[B,N_t,C_s+C_skip]` | `[B,N_t,C_out]` |
| **FP4** | `[B,16,512]` | `[B,64,256]` | `[B,64,512]` | `[B,64,768]` | `[B,64,256]` |
| **FP3** | `[B,64,256]` | `[B,256,128]` | `[B,256,256]` | `[B,256,384]` | `[B,256,256]` |
| **FP2** | `[B,256,256]` | `[B,1024,64]` | `[B,1024,256]` | `[B,1024,320]` | `[B,1024,128]` |
| **FP1** | `[B,1024,128]` | `[B,N,C]` | `[B,N,128]` | `[B,N,128+C]` | `[B,N,128]` |
| **Classifier** | – | – | – | – | `[B,N,num_classes]` |



where:
- B : batch size  
- N : number of input points  
- N_s : number of source (sparser) points  
- N_t : number of target (denser) points  
- C_s : source feature channels  
- C_skip : skip connection feature channels  
- C_out : output feature channels





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
Classes are ordered by decreasing frequency in the dataset.


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
<br>All experiments use class-aware sampler. Values are reported as train / validation. Bold values indicate the best validation score. Classes are ordered by decreasing frequency in the dataset.

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
| **Best validation** |||||||
| Best mIoU | ----- / 0.816 | ----- / **0.816** | ----- / 0.815 | ----- / **0.816** | ----- / 0.806 | ----- / 0.797 |
| Best Loss | ----- / 0.113 | ----- / **0.113** | ----- / 0.114 | ----- / 0.115 | ----- / 0.117 | ----- / 0.124 |
| Best Accuracy | ----- / 0.959 | ----- / **0.959** | ----- / **0.959** | ----- / 0.958 | ----- / 0.957 | ----- / 0.954 |


**Table 3**. Comparison of PointNet++ configurations: NLL loss + near-uniform class weighting [0.9894, 0.9894, 0.9894, 1.0049, 1.0270]
<br>All experiments use class-aware sampler. Values are reported as train / validation. Bold values indicate the best validation score. Classes are ordered by decreasing frequency in the dataset.



### FINAL CONFIG (optimal for PoinNet)

| Metric | 1 - baseline | 2 - dropout | 3 - K-neighbors | 4 - ball_closest | 5 - ball_random | 6 - xyz only |
|------|------|------|------|------|------|------|
| **Overall metrics** |||||||
| mIoU | 0.810 / 0.800 | 0.816 / **0.806** | 0.811 / 0.806 | 0.805 / 0.801 | 0.802 / 0.798 |  |
| Loss | 0.127 / 0.129 | 0.122 / 0.127 | 0.126 / 0.127 | 0.131 / 0.129 | 0.133 / 0.132 |  |
| Accuracy | 0.959 / 0.957 | 0.960 / 0.957 | 0.959 / 0.957 | 0.957 / 0.956 | 0.957 / 0.956 |  |
| **Class IoU** |||||||
| Ground | 0.951 / 0.946 | 0.952 / 0.946 | 0.951 / 0.946 | 0.950 / 0.945 | 0.949 / 0.945 |  |
| Vegetation | 0.868 / 0.864 | 0.871 / **0.866** | 0.868 / 0.865 | 0.865 / 0.864 | 0.864 / 0.863 |  |
| Buildings | 0.954 / 0.954 | 0.955 / **0.955** | 0.954 / **0.955** | 0.948 / 0.950 | 0.948 / 0.950 |  |
| Vehicle | 0.686 / 0.653 | 0.699 / 0.659 | 0.688 / **0.666** | 0.675 / 0.657 | 0.668 / 0.648 |  |
| Utility | 0.594 / 0.585 | 0.603 / **0.602** | 0.593 / 0.596 | 0.584 / 0.586 | 0.581 / 0.584 |  |
| **Best validation** |||||||
| Best mIoU | ----- / 0.806 | ----- / **0.810** | ----- / 0.809 | ----- / 0.801 | ----- / 0.800 |  |
| Best Loss | ----- / 0.126 | ----- / **0.125** | ----- / **0.125** | ----- / 0.129 | ----- / 0.131 |  |
| Best Accuracy | ----- / 0.957 | ----- / **0.958** | ----- / **0.958** | ----- / 0.956 | ----- / 0.956 |  |


<!-- | Metric | 1 - baseline | 2 - dropout | 3 - K-neighbors | 4 - ball_closest | 5 - ball_random | 6 - xyz only |
|------|------|------|------|------|------|------|
| **Overall metrics** |||||||
| mIoU | 0.812 / 0.804 | 0.814 / **0.805** | 0.811 / 0.804 | 0.805 / 0.799 | 0.803 / 0.796 | 0.796 / 0.782 |
| Loss | 0.128 / 0.126 | 0.125 / **0.125** | 0.128 / 0.126 | 0.133 / 0.129 | 0.135 / 0.131 | 0.140 / 0.139 |
| Accuracy | 0.958 / **0.957** | 0.959 / **0.957** | 0.958 / **0.957** | 0.957 / 0.956 | 0.956 / 0.956 | 0.955 / 0.952 |
| **Class IoU** |||||||
| Ground | 0.950 / **0.946** | 0.951 / **0.946** | 0.950 / **0.946** | 0.950 / 0.945 | 0.949 / 0.945 | 0.945 / 0.940 |
| Vegetation | 0.866 / 0.865 | 0.867 / **0.866** | 0.866 / **0.866** | 0.863 / 0.863 | 0.862 / 0.862 | 0.856 / 0.852 |
| Buildings | 0.954 / **0.955** | 0.954 / **0.955** | 0.953 / **0.955** | 0.948 / 0.950 | 0.947 / 0.950 | 0.949 / 0.950 |
| Vehicle | 0.686 / 0.667 | 0.691 / **0.669** | 0.684 / 0.665 | 0.675 / 0.653 | 0.670 / 0.648 | 0.653 / 0.609 |
| Utility | 0.605 / 0.588 | 0.607 / **0.589** | 0.602 / **0.589** | 0.590 / 0.581 | 0.586 / 0.574 | 0.577 / 0.561 |
| **Best validation** |||||||
| Best mIoU | ----- / 0.806 | ----- **0.810** | ----- 0.807 | ----- 0.800 | ----- 0.801 | ----- 0.791 |
| Best Loss | ----- / 0.126 | ----- **0.125** | ----- 0.126 | ----- 0.129 | ----- 0.131 | ----- 0.137 |
| Best Accuracy | ----- 0.957 | ----- **0.958** | ----- 0.957 | ----- 0.956 | ----- 0.956 | ----- 0.953 | -->




TODO!!!


Table X summarizes the performance of the evaluated PointNet++ configurations. The baseline model achieves strong overall performance, with a validation mIoU of 0.804 and balanced results across most classes. Introducing dropout slightly improves the best validation mIoU (0.810) while maintaining similar class-wise performance, suggesting a modest regularization benefit. The K-neighbors configuration yields results comparable to the baseline, indicating that the neighborhood selection strategy has limited impact in this setting. In contrast, the ball-based grouping variants slightly degrade performance, particularly for smaller classes such as Vehicle and Utility. Finally, removing additional input features and using only XYZ coordinates leads to the lowest performance across all metrics, highlighting the importance of feature information beyond spatial coordinates.


The ball-based grouping strategies slightly degrade performance, particularly for smaller classes such as Vehicle and Utility. This behaviour is likely related to the variable number of neighbours produced by radius-based grouping in sparse regions, which can lead to unstable feature aggregation compared to the fixed-size neighbourhoods provided by k-nearest neighbours.

The idea was to make neighbourhoods geometrically consistent, rather than point-count consistent like KNN.

Increasing the neighborhood size slightly improves the IoU for the Vehicle class. Vehicles are relatively small and sparse objects in the scene, and a larger neighborhood provides additional contextual information that may help distinguish them from surrounding structures.
However, Even though Vehicle IoU improves, the overall mIoU does not. 

Whereas larger context helps with vehicles, it hurts for utilities. Since utility structures are much smaller, they contain very few points, tehrefore, increasing the number of neighbours quickly inccreases the number of points from other classes, making the context less pure. So there is a tradeoff between the size of teh structure to identify and the size of the neighbourhood,  what seems to provide useful semantinc context very easily can hurt more than help. 


PointNet++ extracts local geometric features by grouping neighbouring points around sampled centroids. Two common strategies for this grouping are k-nearest neighbours (KNN) and radius-based (ball) query, which differ in how the local neighbourhood is defined.

In KNN grouping, the algorithm always selects a fixed number K of neighbouring points. This guarantees that each neighbourhood contains the same number of points, providing stable input to the neural network. However, the spatial size of the neighbourhood can vary depending on point density. For example, if K=32, the 32 nearest points in a dense vegetation region may lie very close to the centroid, while the same 32 points in a sparse area may cover a much larger spatial region.

In contrast, ball query grouping selects all points within a fixed spatial radius r. This ensures that the neighbourhood always corresponds to the same physical scale. For instance, if 
r=0.2 m, the network always analyzes the geometry within that spatial extent. In other words, the network learns features such as “what does the geometry look like inside a 20 cm region?” rather than “what do the nearest 32 points look like?”. For this reason, ball query is often considered more invariant to changes in point density.

In practice, however, the number of points within a fixed radius can vary significantly across the scene. Dense regions may contain many points inside the radius, while sparse regions may contain only a few. This can lead to unstable feature aggregation, especially for small or sparsely sampled objects. In our experiments, the ball-based grouping strategies consistently produced slightly worse results, particularly for the Vehicle and Utility classes. These objects contain relatively few points, meaning that the fixed-radius neighbourhood often provides limited local information. By contrast, KNN grouping guarantees that each centroid always receives the same number of neighbouring points, which results in more stable feature extraction.

Overall, while ball query enforces a consistent spatial scale for neighbourhoods, the results suggest that KNN grouping provides more robust performance for this dataset, especially when dealing with small or sparsely sampled objects.


Ball query is worse here because:

neighbourhood size varies

sparse classes get very few neighbours

KNN gives stable feature extraction

"knn" vs "ball_closest"/"ball_random" is not a fair comparison 


knn is robust in sparse areas (always enough points), but spatial neighborhood varies a lot
ball has a consistent geometric scale (bounding), but may contain very few points.
If very dense areas:
ball_closest → biased toward center 
ball_random  → more spatially uniform sampling


### RESULTS:
