import matplotlib.pyplot as plt
import numpy as np


def visualize_point_cloud_on_axis(ax, point_cloud, label="Point Cloud"):
    """Visualize a 3D point cloud on a given axis (for subplots).

    Args:
        ax: matplotlib 3D axis to plot on
        point_cloud: numpy array of shape (N, 3) with x, y, z coordinates
        label: title for the plot
    """

    ax.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1],
        point_cloud[:, 2],
        marker="o",
        s=1,
    )

    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.set_xlabel("X", fontsize=6)
    ax.set_ylabel("Y", fontsize=6)
    ax.set_zlabel("Z", fontsize=6)

    # Adjust viewing angle for better visualization
    ax.view_init(elev=20, azim=45)

    # Remove tick labels to reduce clutter in grid view
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def visualize_point_cloud_matplotlib(point_cloud, label="Point Cloud", figsize=(10, 8), output_path=None):
    """Visualize a 3D point cloud with better color representation.

    Args:
        point_cloud: numpy array of shape (N, 3) with x, y, z coordinates
        label: title for the plot
        figsize: size of the figure
        output_path: path to save the figure (if None, will display)
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    visualize_point_cloud_on_axis(ax=ax, point_cloud=point_cloud, label=label)
    # plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return


def visualize_multiple_point_clouds(point_clouds, labels=None, n_cols=3, figsize=(15, 10), output_path=None, title="Point Clouds Visualization"):
    """Visualize multiple 3D point clouds in a grid layout.

    Args:
        point_clouds: list of numpy arrays, each of shape (N, 3)
        labels: list of titles for each plot
        n_cols: number of columns in the grid
        figsize: size of the figure
        output_path: path to save the figure (if None, will display)
        title: overall title for the figure
    """
    n_points = len(point_clouds)
    n_rows = int(np.ceil(n_points / n_cols))

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for i, point_cloud in enumerate(point_clouds):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        label = labels[i] if labels and i < len(
            labels) else f"Point Cloud {i+1}"
        visualize_point_cloud_on_axis(
            ax=ax, point_cloud=point_cloud, label=label)

    # plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return
