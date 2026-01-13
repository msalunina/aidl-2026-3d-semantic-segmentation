"""
3D LAS File Visualization using Plotly
Interactive 3D visualization with proper legends and class coloring
"""

import numpy as np
import laspy
import plotly.graph_objects as go
from pathlib import Path
import random


# TerLiDAR dataset class definitions
CLASS_NAMES = {
    0: 'Never Classified',
    1: 'Unclassified',
    2: 'Ground',
    3: 'Low Vegetation',
    4: 'Medium Vegetation',
    5: 'High Vegetation',
    6: 'Building',
    7: 'Low Points',
    8: 'Ground Key Points',
    9: 'Water',
    10: 'Rail',
    11: 'Road Surface',
    12: 'Wire - Guard',
    13: 'Wire - Conductor',
    14: 'Transmission Tower',
    15: 'Wire - Connector',
    16: 'Bridge Deck',
    17: 'High Noise'
}

# Color mapping for each class (matching exactly plot_las_visualization.py)
CLASS_COLORS = {
    0: 'lightgray',      # Never Classified (default)
    1: 'brown',          # Unclassified
    2: 'tan',            # Ground
    3: 'lightgreen',     # Low Vegetation
    4: 'forestgreen',    # Medium Vegetation
    5: 'darkgreen',      # High Vegetation
    6: 'orange',         # Building
    7: 'black',          # Low Points
    8: 'yellow',         # Ground Key Points
    9: 'blue',           # Water
    10: 'silver',        # Rail
    11: 'pink',          # Road Surface / Air Points
    12: 'cyan',          # Wire - Guard
    13: 'cyan',          # Other Ground
    14: 'blue',          # Power Lines
    15: 'purple',        # Transmission Towers
    16: 'red',           # Wall
    17: 'magenta',       # Above Building
    18: 'navy',          # Other Towers
    135: 'gray',         # Noise
}


def subsample_points(points, labels, max_points=100000, method='random'):
    """
    Subsample points to a maximum number
    
    Parameters:
    -----------
    points : np.ndarray
        Array of shape (N, 3) with XYZ coordinates
    labels : np.ndarray
        Array of shape (N,) with class labels
    max_points : int
        Maximum number of points to keep
    method : str
        Sampling method: 'random', 'uniform', or 'stratified'
    
    Returns:
    --------
    subsampled_points : np.ndarray
        Subsampled points
    subsampled_labels : np.ndarray
        Subsampled labels
    """
    n_points = len(points)
    
    if n_points <= max_points:
        return points, labels
    
    if method == 'random':
        # Simple random sampling
        indices = np.random.choice(n_points, max_points, replace=False)
    
    elif method == 'uniform':
        # Uniform sampling (every nth point)
        step = n_points // max_points
        indices = np.arange(0, n_points, step)[:max_points]
    
    elif method == 'stratified':
        # Stratified sampling - keep class distribution
        unique_classes = np.unique(labels)
        indices = []
        
        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]
            n_cls = len(cls_indices)
            n_sample = max(1, int(max_points * n_cls / n_points))
            
            if n_sample < n_cls:
                sampled = np.random.choice(cls_indices, n_sample, replace=False)
            else:
                sampled = cls_indices
            
            indices.extend(sampled.tolist())
        
        indices = np.array(indices)
        # If we have too many, randomly subsample to exact number
        if len(indices) > max_points:
            indices = np.random.choice(indices, max_points, replace=False)
    
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    return points[indices], labels[indices]


def plot_las_3d_plotly(las_file_path, max_points=100000, sampling_method='stratified',
                       color_by='class', show_legend=True, point_size=2, 
                       opacity=0.8, title=None, height=800, width=1200,
                       save_html=None, show_stats=True):
    """
    Create an interactive 3D visualization of a LAS file using Plotly
    
    Parameters:
    -----------
    las_file_path : str or Path
        Path to the .las file to visualize
    max_points : int, default=100000
        Maximum number of points to display (for performance)
    sampling_method : str, default='stratified'
        Subsampling method: 'random', 'uniform', or 'stratified'
    color_by : str, default='class'
        How to color points: 'class', 'height', 'intensity', or 'rgb'
    show_legend : bool, default=True
        Whether to display the legend with class names and counts
    point_size : int, default=2
        Size of points in the plot
    opacity : float, default=0.8
        Opacity of points (0=transparent, 1=opaque)
    title : str, optional
        Custom title for the plot
    height : int, default=800
        Height of the plot in pixels
    width : int, default=1200
        Width of the plot in pixels
    save_html : str or Path, optional
        Path to save the interactive HTML plot
    show_stats : bool, default=True
        Print statistics about the point cloud
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The Plotly figure object
    
    Examples:
    ---------
    >>> # Basic usage
    >>> fig = plot_las_3d_plotly('data_red/434652_ICGC.las')
    >>> fig.show()
    
    >>> # With custom parameters
    >>> fig = plot_las_3d_plotly('data.las', max_points=50000, 
    ...                          sampling_method='random', point_size=3)
    
    >>> # Save as HTML
    >>> fig = plot_las_3d_plotly('data.las', save_html='output.html')
    """
    las_file_path = Path(las_file_path)
    
    if not las_file_path.exists():
        raise FileNotFoundError(f"LAS file not found: {las_file_path}")
    
    # Read the LAS file
    print(f"Loading LAS file: {las_file_path.name}")
    las_data = laspy.read(las_file_path)
    
    # Extract coordinates
    x = np.array(las_data.x, dtype=np.float32)
    y = np.array(las_data.y, dtype=np.float32)
    z = np.array(las_data.z, dtype=np.float32)
    points = np.stack([x, y, z], axis=1)
    
    # Extract classification labels
    if hasattr(las_data, 'classification'):
        labels = np.array(las_data.classification, dtype=np.int32)
    else:
        labels = np.zeros(len(points), dtype=np.int32)
        print("Warning: No classification found in LAS file")
    
    if show_stats:
        print(f"Total points: {len(points):,}")
        unique_classes = np.unique(labels)
        print(f"Unique classes: {sorted(unique_classes.tolist())}")
    
    # Subsample if necessary
    if len(points) > max_points:
        print(f"Subsampling from {len(points):,} to {max_points:,} points using '{sampling_method}' method")
        points, labels = subsample_points(points, labels, max_points, sampling_method)
    
    # Determine colors
    if color_by == 'class':
        colors, hover_text, legend_groups = _color_by_class(points, labels, las_file_path.name)
    elif color_by == 'height':
        colors, hover_text, legend_groups = _color_by_height(points, labels)
    elif color_by == 'intensity':
        colors, hover_text, legend_groups = _color_by_intensity(points, labels, las_data)
    elif color_by == 'rgb':
        colors, hover_text, legend_groups = _color_by_rgb(points, labels, las_data)
    else:
        raise ValueError(f"Unknown color_by option: {color_by}")
    
    # Create the 3D scatter plot
    fig = go.Figure()
    
    if show_legend and color_by == 'class':
        # Create separate traces for each class (for proper legend)
        for class_id, class_info in legend_groups.items():
            mask = labels == class_id
            if np.sum(mask) == 0:
                continue
            
            fig.add_trace(go.Scatter3d(
                x=points[mask, 0],
                y=points[mask, 1],
                z=points[mask, 2],
                mode='markers',
                name=class_info['name'],
                marker=dict(
                    size=point_size,
                    color=class_info['color'],
                    opacity=opacity,
                ),
                hovertext=[hover_text[i] for i in np.where(mask)[0]],
                hoverinfo='text',
                showlegend=True
            ))
    else:
        # Single trace without legend
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=colors,
                opacity=opacity,
                colorscale='Viridis' if color_by == 'height' else None,
                showscale=color_by in ['height', 'intensity'],
                colorbar=dict(title='Height (Z)' if color_by == 'height' else 'Intensity') if color_by in ['height', 'intensity'] else None
            ),
            hovertext=hover_text,
            hoverinfo='text',
            showlegend=False
        ))
    
    # Set layout
    if title is None:
        title = f"3D Point Cloud: {las_file_path.name}"
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, family='Arial Black')
        ),
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Z (meters)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=height,
        width=width,
        hovermode='closest',
        showlegend=show_legend,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="Black",
            borderwidth=2,
            font=dict(size=11),  # Increase font size for readability
        )
    )
    
    # Save as HTML if requested
    if save_html:
        save_html = Path(save_html)
        save_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_html))
        print(f"Interactive plot saved to: {save_html}")
    
    return fig


def _color_by_class(points, labels, filename):
    """Color points by classification class"""
    unique_classes = np.unique(labels)
    n_points = len(points)
    
    # Count points per class
    class_counts = {cls: np.sum(labels == cls) for cls in unique_classes}
    
    # Create colors and hover text
    colors = []
    hover_text = []
    legend_groups = {}
    
    for i, (point, label) in enumerate(zip(points, labels)):
        class_id = int(label)
        class_name = CLASS_NAMES.get(class_id, f'Class {class_id}')
        color = CLASS_COLORS.get(class_id, 'gray')
        
        colors.append(color)
        hover_text.append(
            f"Class: {class_id} - {class_name}<br>"
            f"X: {point[0]:.2f}<br>"
            f"Y: {point[1]:.2f}<br>"
            f"Z: {point[2]:.2f}"
        )
        
        # Store legend info
        if class_id not in legend_groups:
            count = class_counts[class_id]
            percentage = (count / n_points) * 100
            legend_groups[class_id] = {
                'name': f'{class_name} ({count:,}, {percentage:.1f}%)',
                'color': color,
                'count': count
            }
    
    return colors, hover_text, legend_groups


def _color_by_height(points, labels):
    """Color points by height (Z coordinate)"""
    z_values = points[:, 2]
    colors = z_values
    
    hover_text = [
        f"Height: {z:.2f}m<br>X: {x:.2f}<br>Y: {y:.2f}"
        for x, y, z in points
    ]
    
    return colors, hover_text, {}


def _color_by_intensity(points, labels, las_data):
    """Color points by intensity values"""
    if not hasattr(las_data, 'intensity'):
        print("Warning: No intensity data found, using height instead")
        return _color_by_height(points, labels)
    
    intensity = np.array(las_data.intensity, dtype=np.float32)
    
    # Normalize intensity
    intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
    
    hover_text = [
        f"Intensity: {int(i)}<br>X: {x:.2f}<br>Y: {y:.2f}<br>Z: {z:.2f}"
        for (x, y, z), i in zip(points, intensity)
    ]
    
    return intensity_norm, hover_text, {}


def _color_by_rgb(points, labels, las_data):
    """Color points by RGB values"""
    if not (hasattr(las_data, 'red') and hasattr(las_data, 'green') and hasattr(las_data, 'blue')):
        print("Warning: No RGB data found, using height instead")
        return _color_by_height(points, labels)
    
    # Extract and normalize RGB
    red = np.array(las_data.red, dtype=np.float32) / 65535.0
    green = np.array(las_data.green, dtype=np.float32) / 65535.0
    blue = np.array(las_data.blue, dtype=np.float32) / 65535.0
    
    # Create RGB color strings
    colors = [
        f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
        for r, g, b in zip(red, green, blue)
    ]
    
    hover_text = [
        f"RGB: ({int(r*255)}, {int(g*255)}, {int(b*255)})<br>"
        f"X: {x:.2f}<br>Y: {y:.2f}<br>Z: {z:.2f}"
        for (x, y, z), r, g, b in zip(points, red, green, blue)
    ]
    
    return colors, hover_text, {}


def plot_multiple_las_files(las_files, max_points_per_file=50000, **kwargs):
    """
    Plot multiple LAS files in the same 3D view
    
    Parameters:
    -----------
    las_files : list of str or Path
        List of paths to LAS files
    max_points_per_file : int
        Maximum points to load from each file
    **kwargs : dict
        Additional arguments passed to plot_las_3d_plotly
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The combined Plotly figure
    """
    fig = go.Figure()
    
    all_points = []
    all_labels = []
    all_filenames = []
    
    for las_file in las_files:
        las_file = Path(las_file)
        las_data = laspy.read(las_file)
        
        # Extract data
        x = np.array(las_data.x, dtype=np.float32)
        y = np.array(las_data.y, dtype=np.float32)
        z = np.array(las_data.z, dtype=np.float32)
        points = np.stack([x, y, z], axis=1)
        
        if hasattr(las_data, 'classification'):
            labels = np.array(las_data.classification, dtype=np.int32)
        else:
            labels = np.zeros(len(points), dtype=np.int32)
        
        # Subsample
        if len(points) > max_points_per_file:
            points, labels = subsample_points(points, labels, max_points_per_file, 'stratified')
        
        all_points.append(points)
        all_labels.append(labels)
        all_filenames.extend([las_file.name] * len(points))
        
        print(f"Loaded {len(points):,} points from {las_file.name}")
    
    # Combine all points
    combined_points = np.vstack(all_points)
    combined_labels = np.concatenate(all_labels)
    
    print(f"\nTotal points: {len(combined_points):,}")
    
    # Create visualization
    colors, hover_text, legend_groups = _color_by_class(
        combined_points, combined_labels, "Multiple Files"
    )
    
    # Add hover text with filename
    for i, fname in enumerate(all_filenames):
        hover_text[i] = f"File: {fname}<br>" + hover_text[i]
    
    # Create traces per class
    for class_id, class_info in legend_groups.items():
        mask = combined_labels == class_id
        if np.sum(mask) == 0:
            continue
        
        fig.add_trace(go.Scatter3d(
            x=combined_points[mask, 0],
            y=combined_points[mask, 1],
            z=combined_points[mask, 2],
            mode='markers',
            name=class_info['name'],
            marker=dict(
                size=kwargs.get('point_size', 2),
                color=class_info['color'],
                opacity=kwargs.get('opacity', 0.8),
            ),
            hovertext=[hover_text[i] for i in np.where(mask)[0]],
            hoverinfo='text',
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title="Combined 3D Point Cloud",
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Z (meters)',
            aspectmode='data'
        ),
        height=kwargs.get('height', 800),
        width=kwargs.get('width', 1200),
        showlegend=True
    )
    
    return fig


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize LAS files in 3D with Plotly')
    parser.add_argument('las_file', type=str, help='Path to LAS file')
    parser.add_argument('--max_points', type=int, default=100000,
                        help='Maximum number of points to display')
    parser.add_argument('--sampling', type=str, default='stratified',
                        choices=['random', 'uniform', 'stratified'],
                        help='Subsampling method')
    parser.add_argument('--color_by', type=str, default='class',
                        choices=['class', 'height', 'intensity', 'rgb'],
                        help='How to color the points')
    parser.add_argument('--point_size', type=int, default=2,
                        help='Size of points')
    parser.add_argument('--save_html', type=str, default=None,
                        help='Save interactive plot to HTML file')
    parser.add_argument('--no_legend', action='store_true',
                        help='Hide the legend')
    
    args = parser.parse_args()
    
    fig = plot_las_3d_plotly(
        las_file_path=args.las_file,
        max_points=args.max_points,
        sampling_method=args.sampling,
        color_by=args.color_by,
        show_legend=not args.no_legend,
        point_size=args.point_size,
        save_html=args.save_html
    )
    
    fig.show()
