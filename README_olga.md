# POINTNET++

## ARCHITECTURE

## Neighborhood definition in PointNet++

In PointNet++, local geometric features are learned by grouping neighboring points around a set of sampled centers obtained via **Farthest Point Sampling (FPS)**.  
The strategy used to select neighbors determines the **spatial support of the local patch**, which directly influences the type of geometric structures the network can capture.

In this work, three grouping strategies are compared:

- **k-nearest neighbors (`kNN`)**
- **radius-based grouping selecting the closest neighbors (`ball_closest`)**
- **radius-based grouping with random sampling (`ball_random`)**

---

### k-Nearest Neighbors (`knn`)

In kNN grouping, for each center point the **K nearest points in Euclidean space** are selected.Therefore:

- **Number of neighbors is fixed**
- **Spatial size of the neighborhood depends on the local point density**
    - In **dense regions**, the K nearest neighbors lie close to the center --> **small spatial patch**.  
    - In **sparse regions**, the same number of neighbors may lie further -->  **much larger spatial area**.

As a result, the **effective receptive field varies with point density**. Although the network always processes the same number of points, 
the **geometric scale represented by those points changes across the scene**, making kNN grouping 
**density-dependent in terms of spatial coverage**.

In deeper **Set Abstraction (SA)** layers, FPS progressively reduces the number of centers, increasing the spacing between them. 
Consequently, the spatial extent of kNN neighborhoods naturally grows with depth, even when the value of **K** remains constant. 
This allows deeper layers to capture **larger geometric structures**, such as buildings or terrain.


### Radius-Based Grouping (Ball Query)

In radius-based grouping, neighbors are selected within a **fixed spatial radius** around each center.

This defines the **maximum spatial support of the neighborhood**, which is controlled by the radius and therefore independent of the 
local point density. However, since PointNet++ expects a fixed number of neighbors, only up to **K neighbors** are retained (PointNet++ expects a fixed number of neighbors), 
therefore, the **effective receptive field may still depend on how those neighbors are selected**, especially in dense regions (more than K points inside the ball)

Two strategies were implemented to select these **K neighbors**.



- `ball_closest`: 

    When more than **K** points lie inside the radius, only the **K closest points to the center** are selected.

    Although the neighborhood is bounded by the radius, this selection strategy introduces a **bias toward points located near the center**. 
    As a consequence, the **effective receptive field may shrink in dense regions**. Therefore:

    - **Maximum spatial support is fixed**
    - Effective receptive field may be smaller than the radius, introducing a **mild density dependence**



- `ball_random`: 

    When more than **K** points lie inside the radius, **K of them are randomly sampled**.

    In this case, the radius directly defines the **spatial extent of the neighborhood**, since any point within the ball has an equal probability 
    of being selected. Therefore:

    - **Maximum spatial support is fixed**
    - Effective receptive field closely matches the radius, making this strategy largely **density independent**


# Summary

| Method | Spatial Support | Effective Receptive Field | Density Dependence |
|------|------|------|------|
| **knn** | Variable | Variable | High |
| **ball_closest** | Fixed (radius) | May shrink in dense regions | Moderate |
| **ball_random** | Fixed (radius) | Approximately equal to radius | Low |

---



This comparison highlights how the **choice of grouping strategy controls the spatial scale of the features learned by PointNet++**, 
and therefore influences the types of geometric structures that can be captured at different levels of the network hierarchy.


## Hierarchical feature learning in PointNet++

PointNet++ builds a hierarchical representation of the point cloud through successive **Set Abstraction (SA)** layers. Each layer applies:

1. **Farthest Point Sampling (FPS)** to select a subset of centers  
2. **Neighborhood grouping** around each center  
3. A **shared MLP + max pooling** to extract local geometric features  

As the network progresses through the hierarchy (towards deeper layers), the number of centers decreases and the distance between them increases.  
Therefore, even if the number of neighbors remains constant, the spatial extent of each neighborhood naturally grows with depth, allowing each 
layer to capture geometric structures at a different spatial scale.

For example:

Early layers focus on **small local structures**, while deeper layers represent **larger semantic objects**.

For instance:

- **SA1** may capture fine geometric patterns such as edges, corners, or small object parts (e.g., parts of a car wheel or tree branches).
- **SA2** may capture slightly larger structures, such as parts of objects (e.g., a full car roof or part of a tree canopy).
- **SA3** may capture complete objects or object groups (e.g., an entire car or a section of vegetation).
- **SA4** may represent larger scene elements, such as a **car within a street segment**, a **group of trees**, or a **portion of terrain**.

In this way, the receptive field grows progressively across layers, allowing the network to transition from **fine geometric features** to 
**high-level semantic structures** describing the scene.



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

| PointNet++|      test     |    grouping    | dropout |  K neighbours |              channels                |
|:---------:|:-------------:|:--------------:|:-------:|:-------------:|:------------------------------------:|
|     1     |  baseline     |      "knn"     |    0.5  | [32,32,32,32] | [xyz,return_number,number_of_returns]|
|     2     | test dropout  |      "knn"     |    0.3  | [32,32,32,32] | [xyz,return_number,number_of_returns]|
|     3     | test K        |      "knn"     |    0.5  | [32,32,64,64] | [xyz,return_number,number_of_returns]|
|     4     | test grouping | "ball_closest" |    0.5  | [32,32,32,32] | [xyz,return_number,number_of_returns]|
|     5     | test grouping | "ball_random"  |    0.5  | [32,32,32,32] | [xyz,return_number,number_of_returns]|
|     6     | test channels |      "knn"     |    0.5  | [32,32,32,32] | [xyz]                                |

### RESULTS:
