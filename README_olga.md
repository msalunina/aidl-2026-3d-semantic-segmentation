
# METRICS: IoU


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



Table 2. Comparison of PointNet++ configurations: Focal loss + class weighting + class-aware sampler (weights: [0.2553, 0.3465, 0.4482, 1.8602, 2.0897])<br>
Values are reported as train / validation. Bold values indicate the best validation score.

| Metric | 1 - baseline<br>olga | 1 - baseline<br>edu | 2 - dropout<br>edu | 3 - K-neighbors<br>olga | 4 - ball_closest<br>edu | 5 - ball_random<br>olga | 6 - xyz only<br>edu |
|------|------|------|------|------|------|------|------|
| mIoU | 0.781 / 0.766 | 0.783 / 0.768 | 0.785 / 0.768 | 0.782 / 0.769 | 0.776 / **0.771** | runnnig | 0.767 / 0.754 |
| Accuracy | 0.953 / 0.951 | 0.954 / 0.951 | 0.954 / 0.950 | 0.954 / **0.952** | 0.952 / **0.952** | - | 0.949 / 0.948 |
| Loss | 0.019 / 0.019 | 0.019 / 0.019 | 0.018 / 0.021 | 0.019 / 0.019 | 0.020 / **0.018** | - | 0.021 / 0.020 |
| IoU Buildings | 0.950 / 0.947 | 0.950 / 0.947 | 0.950 / 0.942 | 0.950 / **0.948** | 0.943 / **0.948** | - | 0.945 / 0.947 |
| IoU Ground | 0.947 / 0.942 | 0.947 / 0.942 | 0.947 / 0.940 | 0.947 / **0.943** | 0.946 / **0.943** | - | 0.941 / 0.938 |
| IoU Utility | 0.537 / 0.526 | 0.543 / **0.551** | 0.544 / 0.541 | 0.538 / 0.541 | 0.531 / 0.515 | - | 0.511 / 0.501 |
| IoU Vegetation | 0.853 / 0.849 | 0.854 / 0.849 | 0.856 / 0.845 | 0.854 / 0.851 | 0.851 / **0.855** | - | 0.844 / 0.843 |
| IoU Vehicle | 0.619 / 0.567 | 0.621 / 0.551 | 0.626 / 0.570 | 0.622 / 0.560 | 0.610 / **0.595** | - | 0.591 / 0.543 |


Table 3. Comparison of PointNet++ configurations: NLL loss + near-uniform class weighting + class-aware sampler (weights: [0.9894, 0.9894, 0.9894, 1.0049, 1.0270])<br>
Values are reported as train / validation. Bold values indicate the best validation score.

| Metric | 1 - baseline| 2 - dropout| 3 - K-neighbors| 4 - ball_closest| 5 - ball_random| 6 - xyz only |
|------|------|------|------|------|------|------|
| mIoU | — | — | — | — | **0.805 / 0.808** | — |
| Accuracy | — | — | — | — | **0.957 / 0.957** | — |
| Loss | — | — | — | — | 0.121 / 0.116 | — |
| IoU Buildings | — | — | — | — | **0.947 / 0.951** | — |
| IoU Ground | — | — | — | — | **0.949 / 0.945** | — |
| IoU Utility | — | — | — | — | **0.597 / 0.605** | — |
| IoU Vegetation | — | — | — | — | **0.864 / 0.867** | — |
| IoU Vehicle | — | — | — | — | **0.671 / 0.670** | — |

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
