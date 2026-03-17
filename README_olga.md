
## Metrics

Evaluating the performance of a semantic segmentation model requires metrics that measure how well the predicted labels match the ground truth labels. In the case of point cloud segmentation, the task consists of assigning a semantic class to every point in the cloud. While classification accuracy measures the proportion of correctly labeled points, it is often not a reliable metric for segmentation tasks because datasets are usually highly imbalanced. For example, in LiDAR datasets large portions of the scene may correspond to dominant classes such as ground or vegetation, while other classes such as vehicles or utilities appear much less frequently.

In such situations, a model could achieve high accuracy simply by predicting the dominant classes, even if it fails to correctly predict rare classes. For this reason, Intersection over Union (IoU) is widely used as a more robust metric for segmentation evaluation.

### Intersection over Union (IoU)
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


### Mean Intersection over Union (mIoU)

In semantic segmentation tasks, IoU is computed independently for each class. To obtain an overall measure of segmentation performance across all classes, the mean Intersection over Union (mIoU) is used.
The mIoU is defined as the average IoU over all classes:

$$
mIoU = \frac{1}{C} \sum_{c=1}^{C} IoU_c
$$

where:
- $C$ is the number of classes
- $IoU_c$ is the IoU for class $c$

By averaging over classes, mIoU ensures that all classes contribute equally to the evaluation, preventing dominant classes from disproportionately influencing the metric. This makes mIoU particularly suitable for segmentation tasks in datasets with class imbalance. Unlike accuracy, which can be dominated by frequent classes, mIoU evaluates segmentation performance independently for each class and therefore provides a more reliable measure of overall segmentation quality.



### IoU Implementation


For each class, the intersection and the union are computed for every batch and accumulated over the entire epoch. Then, their ratio gives a single IoU value per class and epoch:

$$
IoU_c = \frac{\sum_{b} intersection_c}{\sum_{b} union_c}
$$

where the sum is perfomed over all batches b in the epoch. 



## PointNet++

PointNet++ is a deep neural network designed to process unordered point sets sampled from a metric space. The architecture extends the original PointNet by introducing a hierarchical feature learning framework that captures both local geometric structures and global contextual information.

While PointNet processes the entire point cloud using a single global aggregation, PointNet++ organizes the computation into multiple levels of abstraction, progressively learning features from small local neighborhoods to larger spatial regions. This hierarchical structure enables the network to capture fine-grained geometric patterns and improves performance on complex scenes and segmentation tasks.

The architecture consists of two main components:

- Encoder (Set Abstraction layers, SA) – extracts hierarchical features from the input point cloud.
- Decoder (Feature Propagation layers, FP) – propagates the learned features back to the original points for dense prediction tasks such as semantic segmentation.

Next Figure shows the PointNet++ architecture 

![PointNet++ architecture](figs/pnpp_architecture.png)
(from "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space" (Qi et al., 2017).)




### Encoder
The encoder builds a hierarchical representation of the point cloud through successive **Set Abstraction (SA)** layers. Each SA layer applies sequentially:

1. **Farthest Point Sampling (FPS)** to select a subset of centers  
2. **Neighborhood grouping** around each center  
3. A **shared MLP + max pooling** to extract local geometric features  (mini PointNet)

As the network progresses towards deeper layers, the number of centers decreases and the distance between them increases. Therefore, even if the number of neighbors remains constant, the spatial extent of each neighborhood (effective receptive field) naturally grows with depth, allowing each layer to capture geometric structures at a different spatial scale. Early layers focus on **small local structures (fine geometric structures)**, while deeper layers represent **larger semantic structures** describing the scene.

#### 1. Farthest Point Sampling (FPS)

In order to select the center points, PointNet++ uses Farthest Point Sampling (FPS). It iteratively selects points that maximize the distance from previously selected centers. This ensures that the sampled points are evenly distributed across the point cloud.


#### 2. Neighborhood Grouping

The grouping stage constructs local neighbourhoods around each center. The strategy used to select neighbors determines the **spatial support of the local patch**, which directly influences the type of geometric structures the network can capture. Two grouping strategies are compared:

- **k-Nearest Neighbors**: selects the K closest points
- **Radius-Based grouping**: selects all points within a predefined radius


##### k-Nearest Neighbors (`knn`)

In knn grouping, for each center point the K nearest points in Euclidean space are selected. Consequently:
- **Number of neighbors** is fixed
- **Spatial size of the neighborhood** depends on the local point density. 

In dense regions, the K nearest neighbors lie close to the center and define a small spatial patch, whereas in sparse regions, the same number of neighbors may lie further defining a much larger spatial area. As a result, the **effective receptive field** varies with point density.

##### Radius-Based Grouping

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



#### 3. A **Shared MLP + Max Pooling**

After selecting the K neighbors for each center, each local region is processed by a mini-PointNet network, which learns a feature representation for the neighborhood. This consists of a shared multilayer perceptron (MLP) applied independently to each point followed by a symmetric aggregation function (max pooling) to obtain a single feature vector representing the region.


### Decoder

While the encoder progressively reduces the number of points and extracts higher-level features, semantic segmentation requires a prediction for **every original input point**. Therefore, PointNet++ includes a decoder composed of successive **Feature Propagation (FP)** layers, which progressively upsample features from sparse point sets back to denser ones. Each Feature Propagation layer applies sequentially:

1. **3-NN interpolation** to transfer features from a sparse set of points to a denser one  
2. **Concatenation with skip features** coming from the encoder  
3. A **shared MLP** to refine the propagated features  

In this way, the decoder combines high-level semantic information from deeper layers with fine geometric details preserved by the early encoder layers.


#### 1. 3-NN Interpolation

At each decoder stage, features from a sparse set of points are interpolated onto a denser set of target points. For each target point, the three nearest source points are identified and their features are interpolated using weights inversely proportional to their distance. Therefore, closer source points contribute more strongly to the interpolated feature, while farther ones contribute less.

This interpolation step allows features learned at coarse spatial resolutions to be transferred back to denser point sets.

#### 2. Concatenation with Skip Features

After interpolation, the propagated features are concatenated with the corresponding **skip features** coming from the encoder.

These skip features provide local geometric information extracted at earlier abstraction levels, where the point resolution is still relatively high. Consequently, the decoder does not rely only on coarse semantic information from deep layers, but also reuses fine-grained spatial details that may have been lost during downsampling.

#### 3. Shared MLP Refinement

Once the interpolated features and skip features have been concatenated, the resulting feature vectors are refined using a **shared multilayer perceptron (MLP)** applied independently to each point. In doing so, semantic information coming from the decoder and geometric details preserved from the skip connections are fused together, producing a more informative point-wise representation

By stacking several Feature Propagation layers, the decoder progressively reconstructs point features at increasing resolutions until features are available for the full original point cloud size.



### Network Architecture 


#### Encoder Architecture (SA Layers)

| Layer | # Centers (FPS) | Neighborhood | K | Radius | MLP |
|:------:|:-----------------:|:-------------:|:---:|:-------:|:------|
| **SA1** | 1024 | knn / ball query | 32 | 0.08 | `[32,32,64]` |
| **SA2** | 256 | knn / ball query | 32 | 0.10 | `[64,64,128]` |
| **SA3** | 64 | knn / ball query | 32 | 0.20 | `[128,128,256]` |
| **SA4** | 16 | knn / ball query | 32 | 0.40 | `[256,256,512]` |

**Tensor Shapes**

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


#### Decoder Architecture (FP layers)

| Layer | Interpolation | Skip connection | MLP |
|:------:|:---------------:|:----------------:|:------|
| **FP4** | 3-NN (16 → 64) | SA3 features | `[256,256]` |
| **FP3** | 3-NN (64 → 256) | SA2 features | `[256,256]` |
| **FP2** | 3-NN (256 → 1024) | SA1 features | `[256,128]` |
| **FP1** | 3-NN (1024 → N) | input features (if any) | `[128,128]` |
| **Classifier head** | – | – | `[128,128,num_classes]` |

_NOTE: In the original PointNet++ semantic segmentation architecture, the final decoder FP layer is [128,128,128,128,num_classes].
In our implementation, instead, this last FP is split between the **FP1** block and a separate **classifier head** to allow dropout to be applied explicitly to the two final layers before the per-point class score prediction.


**Tensor Shapes**

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





### Experiments

The idea behind this part of the report is to analyse the effect that some hyperparameters have on the PointNet++ newtwork.
As a starting point, we take advantage of the experiments performed on PoinNet and keep the same already optimized choices: 
- Data augmentation (only rotation)
- Class-aware sampler
- NLL loss
- input channels `[xyz,return_number,number_of_returns]`

#### Hypothesis

Then, we want to test four modifications of the baseline configuration: dropout rate, neighborhood size, grouping strategy, and input feature channels. Each experiment isolates one component while keeping the rest of the architecture unchanged.

- **dropout**: The original PointNet++ paper applies a dropout rate of `0.5` in the last two fully connected layers before per-point classification. Dropout acts as a regularization technique to reduce overfitting, but too high dropout rates may lead to underfitting by limiting the network’s capacity to learn meaningful feature representations. In experiment 2, the dropout rate is reduced to `0.3`, following the same modification applied in PointNet. We expect this change to improve performance.

- **K-neighbors**: PointNet++ builds local geometric features by aggregating information from neighboring points around sampled centers. The number of neighbors determines the amount of local context available to the network. In Experiment 3, the neighborhood size is increased in the deeper layers from `[32,32,32,32]` to `[32,32,64,64]`. Larger neighborhoods may provide additional context, which could help the network to better recognize small or sparse objects such as vehicles. However, excessively large neighborhoods may also introduce points from different classes, potentially reducing the purity of the local geometric representation. This can be very damaging for structures with very few points, like Utilities.

- **grouping strategy**: although knn guarantees a fixed number of points per neighborhood, it allows the spatial extent of the neighborhood to vary depending on point density. In contrast, ball query uses a fixed spatial radius, ensuring a consistent geometric scale for feature extraction. Experiments 4 and 5 we compare to two ball-based strategies (`ball_closest` and `ball_random`) with the density-dependent neighborhood (`knn`). We expect ball-based grouping to be more invariant to point density due to the fixed radius. However, in sparse regions the radius may contain very few points, which can reduce the quality of the extracted features.

- **input feature channels**: PointNet++ typically uses both spatial coordinates and additional input features. In experiment 6, only the spatial coordinates (`xyz`) are used as input. We expect to observe a decrease in performance due to the reduced input information.


| Experiment|  what to test |    grouping    |    dropout   |      K-neighbors      |              feature channels          | 
|:---------:|:-------------:|:--------------:|:------------:|:-----------------------:|:------------------------------------:|
|     1     |  baseline     |      `knn`     |      0.5     | [32,32,32,32] (exact K) | [xyz,return_number,number_of_returns]|
|     2     |    dropout    |      `knn`     |      0.3     | [32,32,32,32] (exact K) | [xyz,return_number,number_of_returns]|
|     3     |  K-neighbors  |      `knn`     |      0.5     | [32,32,64,64] (exact K) | [xyz,return_number,number_of_returns]| 
|     4     |  grouping     | `ball_closest` |      0.5     | [32,32,32,32] (max K)   | [xyz,return_number,number_of_returns]| 
|     5     |  grouping     | `ball_random`  |      0.5     | [32,32,32,32] (max K)   | [xyz,return_number,number_of_returns]| 
|     6     |  channels     |      `knn`     |      0.5     | [32,32,32,32] (exact K) | [xyz]                                | 

**Table 1**. Summary of the PointNet++ experiment configurations. Each experiment modifies a specific component of the baseline model: the dropout rate, the number of neighbors, the grouping strategy or the input feature channels. (_Note: for `ball_closest` and `ball_random` grouping strategies, the parameter K does not define the exact number of neighbors but the maximum number that can be selected within the ball. Therefore, the effective size of the neighborhood is in this case controlled by the radius parameter, which is fixed in all experiments to [0.08, 0.1, 0.2, 0.4]. Such values were selected based on preliminary exploratory tests._)


Morover, unlike Pointnet which processes the entire point cloud using a single global aggregation, PointNet++ learns features from small local neighborhoods to larger spatial regions. Consequently, it is by nature more aware of the different sizes of the structures present on a scene and, therefore, it is likely to be less affected by class imblance produced by very small and rare objects. To test this, we will perform the 6-set of experiments twice: 

- A. NLL weighted (PointNet weights: [0.5272, 0.5272, 0.5276, 1.5454, 1.8727])
- B. NLL unweighted (uniform weights: [1.0000, 1.0000, 1.0000, 1.0000, 1.0000])


#### Results

Tables A and B summarize the performance of the evaluated PointNet++ configurations with a weighted and unweighted NLL loss. Overall, the different configurations produce relatively similar results, indicating that the baseline is already well tuned. However, several trends can be observed when modifying specific components of the architecture and remain largely consistent in both settings.


##### A. NLL Weighted Loss (Best for PointNet)  

| Metric<br>NLL weighted  | 1 - baseline | 2 - dropout | 3 - K-neighbors | 4 - ball_closest | 5 - ball_random | 6 - xyz only |
|:------|:------:|:------:|:------:|:------:|:------:|:------:|
| **Overall metrics** |||||||
| mIoU     | 0.810 / 0.800     | 0.816 / **0.806** | 0.811 / **0.806** | 0.805 / 0.801 | 0.802 / 0.798 | 0.795 / 0.788 |
| Loss     | 0.127 / 0.129     | 0.122 / **0.127** | 0.126 / **0.127** | 0.131 / 0.129 | 0.133 / 0.132 | 0.138 / 0.139 |
| Accuracy | 0.959 / **0.957** | 0.960 / **0.957** | 0.959 / **0.957** | 0.957 / 0.956 | 0.957 / 0.956 | 0.955 / 0.953 |
| **Class IoU** |||||||
| Ground     | 0.951 / **0.946** | 0.952 / **0.946** | 0.951 / **0.946** | 0.950 / 0.945 | 0.949 / 0.945 | 0.946 / 0.940 |
| Vegetation | 0.868 / 0.864     | 0.871 / **0.866** | 0.868 / 0.865     | 0.865 / 0.864 | 0.864 / 0.863 | 0.859 / 0.855 |
| Buildings  | 0.954 / 0.954     | 0.955 / **0.955** | 0.954 / **0.955** | 0.948 / 0.950 | 0.948 / 0.950 | 0.950 / 0.949 |
| Vehicle    | 0.686 / 0.653     | 0.699 / 0.659     | 0.688 / **0.666** | 0.675 / 0.657 | 0.668 / 0.648 | 0.652 / 0.632 |
| Utility    | 0.594 / 0.585     | 0.603 / **0.602** | 0.593 / 0.596     | 0.584 / 0.586 | 0.581 / 0.584 | 0.570 / 0.563 |
| **Best validation** |||||||
| Best mIoU     | ----- / 0.806 | ----- / **0.810** | ----- / 0.809     | ----- / 0.801 | ----- / 0.800 | ----- / 0.789 |
| Best Loss     | ----- / 0.126 | ----- / **0.125** | ----- / **0.125** | ----- / 0.129 | ----- / 0.131 | ----- / 0.139 |
| Best Accuracy | ----- / 0.957 | ----- / **0.958** | ----- / **0.958** | ----- / 0.956 | ----- / 0.956 | ----- / 0.953 |

**Table A**. Comparison of PointNet++ configurations: NLL loss + moderate class weighting [0.5272, 0.5272, 0.5276, 1.5454, 1.8727]. Last epoch values are reported as train / validation. Bold values indicate the best validation score. Classes are ordered by decreasing frequency in the dataset. Best validation values refer to the best value achieved during the entire training.




##### B. NLL Unweighted Loss

| Metric<br>NLL unweighted | 1 - baseline | 2 - dropout | 3 - K-neighbors | 4 - ball_closest | 5 - ball_random | 6 - xyz only |   BEST (2+3) |
|:------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| **Overall metrics** ||||||||
| mIoU     | 0.815 / 0.812     | 0.819 / **0.813** | 0.816 / 0.812     | 0.808 / 0.810     | 0.809 / 0.809 | 0.797 / 0.797 | 0.819 / **0.813** |
| Loss     | 0.112 / 0.115     | 0.110 / **0.114** | 0.112 / **0.114** | 0.117 / 0.116     | 0.117 / 0.116 | 0.122 / 0.123 | 0.110 / 0.115     |
| Accuracy | 0.960 / **0.958** | 0.961 / **0.958** | 0.960 / **0.958** | 0.959 / **0.958** | 0.959 / 0.957 | 0.956 / 0.954 | 0.961 / **0.958** |
| **Class IoU** ||||||||
| Ground     | 0.951 / **0.946** | 0.952 / **0.946** | 0.952 / **0.946** | 0.950 / **0.946** | 0.950 / 0.945 | 0.946 / 0.940 | 0.952 / **0.946** |
| Vegetation | 0.871 / 0.868     | 0.873 / 0.868     | 0.871 / **0.869** | 0.868 / 0.868     | 0.868 / 0.867 | 0.861 / 0.858 | 0.873 / 0.868     |
| Buildings  | 0.954 / **0.955** | 0.954 / 0.954     | 0.953 / **0.955** | 0.948 / 0.950     | 0.949 / 0.951 | 0.950 / 0.950 | 0.955 / 0.953     |
| Vehicle    | 0.690 / 0.672     | 0.700 / 0.677     | 0.695 / 0.677     | 0.682 / **0.679** | 0.683 / 0.673 | 0.653 / 0.647 | 0.698 / 0.677     |
| Utility    | 0.611 / **0.620** | 0.616 / 0.618     | 0.608 / 0.616     | 0.594 / 0.609     | 0.594 / 0.609 | 0.574 / 0.590 | 0.616 / `0.621` |
| **Best validation** ||||||||
| Best mIoU     | ----- / 0.815     | ----- / **0.816** | ----- / **0.816** | ----- / 0.810 | ----- / 0.809 | ----- / 0.797 | ----- / `0.818` |
| Best Loss     | ----- / 0.113     | ----- / 0.113     | ----- / **0.112** | ----- / 0.115 | ----- / 0.116 | ----- / 0.123 | ----- / 0.112     |
| Best Accuracy | ----- / **0.959** | ----- / **0.959** | ----- / **0.959** | ----- / 0.958 | ----- / 0.957 | ----- / 0.954 | ----- / **0.959** |

**Table B**. Comparison of PointNet++ configurations: NLL loss + uniform weights (i.e no weights). Last epoch values are reported as train / validation. Bold values indicate the best validation score. Classes are ordered by decreasing frequency in the dataset. Best validation values refer to the best value achieved during the entire training.

#### Discussion

The following figures show the learning curves for the baseline case for both the unweighted and weighted NLL loss. Both configurations follow a very similar training behaviour: the training loss decreases in both cases and the validation curves stabilize after approximately 30 epochs. However, the unweighted case consistently achieves better validation performance, not only regarding the smaller oscilations it exhibits, but also in the smaller loss values.

![Baseline comparison](figs/pnpp_baseline_loss.png)


This behaviour can also be observed in the mIoU curves below, where the unweighted configuration converges to a slightly higher validation mIoU than the weighted version. 

![Baseline comparison](figs/pnpp_baseline_miou.png)
![Baseline comparison](figs/pnpp_baseline_classes.png)

Regarding class IoU, curves indicate that for frequent classes such as Ground, Vegetation, and Buildings, both configurations behave almost identically. However, for rare classes like Vehicle and Utility, the weighted loss does not provide the expected improvement. In fact, the unweighted configuration slightly outperforms the weighted one in the final epochs.

These results suggest that PointNet++ already handles class imbalance reasonably well through its hierarchical architecture, which captures geometric structures at multiple spatial scales. In contrast to PointNet, where strong class weighting was beneficial, the same weighting scheme appears to slightly degrade performance in the PointNet++ setting, specially for rare classes such as Vehicle and Utility.

Regarding the changes with respect to the baseline, frequent classes like Ground, Vegetation and Building behave similar for all experiments showing almost identical IoU values. Rares classes like Vehicle and Utility are the ones that show more variability. Although such variability is not uniform among experiments, their overall mIoU values indicate two clear benefitial modifications: dropout (experiment 2) and number of neighbours (experiment 3). 

##### Effect of Dropout

Reducing the dropout rate from 0.5 to 0.3 produces a small but consistent improvement across most metrics. In both weighted and unweighted settings, the dropout configuration achieves the best validation mIoU (0.816) and slightly better performance for several classes. This impacts in one of the best mIoU values.

##### Effect of Neighborhood Size (K-neighbors)

Experiment 3 increases the number of neighbors in the deeper abstraction layers from [32,32,32,32] to [32,32,64,64]. The resulting performance is slightly higher than the baseline across most metrics. 

One noticeable effect is a slight improvement of the IoU for the Vehicle class. Vehicles are relatively small and sparse objects in the scene, and a larger neighborhood allows the network to capture a larger spatial context around them. This additional context can help distinguish vehicles from surrounding structures. In contrast, results show that the Utility class does not benefit from larger neighborhoods. Utility objects such as poles and street lights are thin vertical structures that contain very few points. Increasing the neighborhood size quickly introduces points from other classes, making the context less pure.

This variability indicates a strong tradeoff between the size of the structure to identify and the size of the neighbourhood providing context, what seems to provide useful semantinc context for one class, may hurt another. 

##### Effect of Grouping Strategy

The knn strategy is robust in sparse areas because it guarantees enough number of points (fixed), which translates into a stable input to the network. However, the spatial size of the neighborhood varies depending on point density. In contrast, ball-based strategies select points within a fixed spatial radius r, which ensures that the neighbourhood always corresponds to the same physical scale. However, may contain very few points depending on the point density. This variability in the number of points can lead to unstable feature aggregation, particularly for small or sparsely sampled classes.

This behaviour is shown in our experiments, the ball-based grouping strategies slightly degrade performance compared to the knn baseline, particularly for the Vehicle and Utility classes which contain relatively few points. 


##### Effect of Input Feature Channels

Experiment 6 evaluates the impact of removing additional input features and using only the XYZ coordinates. As expected, this configuration consistently produces the lowest performance across all metrics, indicating the importance of includding non-geometric features if available. This effect is particularly visible for the Vehicle and Utility classes, which already contain relatively few points. Without the additional feature channels, the model has less information to separate these objects from surrounding structures.


#### Final Model

Overall, the experiments suggest that the baseline PointNet++ configuration is already close to optimal for this dataset. Among the tested configurations, reducing the dropout rate (Experiment 2) and increasing the neighborhood size (Experiment 3) produced the most consistent improvements over the baseline. To evaluate whether both improvements could be combined, a final experiment was performed using both modifications simultaneously. This configuration achieved the highest overall performance, reaching a best validation mIoU of 0.818, slightly outperforming the individual experiments.

However, the improvements are not uniform across all classes. While the combined model improves the IoU of some classes (e.g., Utility), other classes show only marginal changes. This suggests that the effects of these hyperparameters are not strictly additive and may interact during training. Based on the overall mIoU, the combined configuration (dropout 0.3 and increased neighborhood size) is selected as the final model and evaluated on the test set.

| model     |     grouping    |    dropout   |      K-neighbors      |              feature channels          | 
|:---------:|:--------------:|:------------:|:-----------------------:|:------------------------------------:|
|   BEST    |      `knn`     |      0.3     | [32,32,64,64] (exact K) | [xyz,return_number,number_of_returns]|

| Metric<br>NLL unweighted | BEST (2+3)|
|:------|:------:
| **Overall metrics** ||||||||
| mIoU          | |
| Loss          | |
| Accuracy      | |
| **Class IoU** | |
| Ground        | |
| Vegetation    | |
| Buildings     | |
| Vehicle       | |
| Utility       | |



