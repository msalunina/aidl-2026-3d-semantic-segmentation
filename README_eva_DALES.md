## DALES DATASET

DALES Dataset

Our primary focus is the real life (not yinthetic) DALES (Dayton Annotated Laser Earth Scan) dataset, a collection of 40 aerial LiDAR scans representing complex outdoor scenes and distributed in standard .las format. It consists of forty tiles, each spanning 0.5 km x 0.5 km. Because each DALES scene contains a very large number of points, the raw point clouds are too large to be processed directly by compact deep learning models such as PointNet. A preprocessing step is therefore required to convert each large scene into smaller training samples. 

To make the dataset manageable, each scene is divided into overlapping spatial blocks of 50 m × 50 m using a sliding-window strategy with a 25 m stride. This overlap helps preserve continuity between neighboring regions and reduces the risk of losing important structures at block boundaries. Blocks containing too few points are discarded, while valid blocks are retained for training and evaluation. 

![Tiling visualization](figs/Tiled_5080_54435_b00246.png)

The grid illustrates how large scenes are decomposed into smaller overlapping bloks used as pointnet samples.

After tiling, each block is randomly sampled to a fixed size of 4096 points. This fixed-size representation is necessary because PointNet expects the same number of input points for every training sample. If a block contains more than 4096 points, a subset is sampled without replacement; if it contains fewer points, sampling is performed with replacement. The sampled block is then normalized by centering its XYZ coordinates and scaling them to a unit sphere, which improves numerical stability during training. 

To reduce class imbalance and improve learning stability, the original 7 DALES labels (Ground, Vegetation, Cars, Trucks. Poles, Power lines. Fences and Buildings) are mapped into 5 semantic classes:  0 – Ground, 1 – Vegetation, 2 – Building, 3 – Vehicle, 4 – Utility and  -1 – Ignore. 

This preprocessing pipeline transforms the original large-scale DALES scenes into standardized point-cloud blocks that can be efficiently used for semantic segmentation experiments with PointNet-based architectures. (./src/convert_las_to_blocks.py). 

After preprocessing, each point-cloud block is stored in NumPy compressed format (.npz). This choice is motivated by efficiency, simplicity, and compatibility with the training pipeline.

![Block visualization](figs/Block_Image.png)

Point cloud block with simplified class labels

The choice of block size (50 m × 50 m) and point count (4096) is driven by a trade-off between geometric context, computational efficiency, and model capacity, particularly for PointNet-based architectures. The block size defines what part of the world the model sees. The point count defines how detailed that view is. A 50 m × 50 m block is large enough to capture both a building and nearby objects such as vehicles, letting the model to learn contextual relationships while preserving local geometric detail. PointNet requires a fixed-size unordered point set, and 4096 is a widely adopted and effective choice.

A key aspect of our workflow is the geospatial validation of DALES samples. Using metadata stored in the .las files (coordinates, projection system), we mapped the dataset into real-world geographic space and visualized it in Google Earth. This allowed us to: verify spatial correctness of the dataset, understand scene context (urban vs rural structures) and validate tiling alignment with real geography.

![Google Earth](figs/Google_Earth.png)


DALES sample tracked down in Google Earth

To validate preprocessing, we implemented visualization tools to visualize the processed point cloud blocks:

Matplotlib 3D visualization (./src/viz_blocks_matplotlib.py)

Open3D interactive visualization (./src/viz_blocks_open3d.py)

    These tools allow inspection of spatial structure, verification of class distributions and debugging preprocessing steps

![Processed block](figs/Block_5.gif)

Processed point cloud block

