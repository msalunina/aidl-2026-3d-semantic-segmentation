"""
Utility script to analyze LAS dataset and determine optimal training parameters
"""

import laspy
import numpy as np
from pathlib import Path
import argparse
from collections import Counter


def analyze_las_dataset(data_dir):
    """Analyze all LAS files in a directory and provide statistics"""
    
    data_path = Path(data_dir)
    las_files = sorted(list(data_path.glob('*.las')))
    
    if len(las_files) == 0:
        print(f"No .las files found in {data_dir}")
        return
    
    print("=" * 80)
    print(f"LAS DATASET ANALYSIS: {data_dir}")
    print("=" * 80)
    
    total_points = 0
    all_classes = []
    has_rgb_list = []
    has_intensity_list = []
    point_counts = []
    
    print(f"\nAnalyzing {len(las_files)} files...\n")
    
    print(f"{'File':<30} {'Points':<12} {'Classes':<10} {'RGB':<6} {'Intensity':<10}")
    print("-" * 80)
    
    for las_file in las_files:
        las = laspy.read(las_file)
        
        n_points = len(las.points)
        point_counts.append(n_points)
        total_points += n_points
        
        # Check for classification
        if hasattr(las, 'classification'):
            unique_classes = np.unique(las.classification)
            all_classes.extend(unique_classes)
            n_classes = len(unique_classes)
        else:
            n_classes = 0
        
        # Check for RGB
        has_rgb = hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue')
        has_rgb_list.append(has_rgb)
        
        # Check for intensity
        has_intensity = hasattr(las, 'intensity')
        has_intensity_list.append(has_intensity)
        
        print(f"{las_file.name:<30} {n_points:<12,} {n_classes:<10} {'Yes' if has_rgb else 'No':<6} {'Yes' if has_intensity else 'No':<10}")
    
    print("=" * 80)
    
    # Summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 80)
    print(f"Total files: {len(las_files)}")
    print(f"Total points: {total_points:,}")
    print(f"Average points per file: {np.mean(point_counts):,.0f}")
    print(f"Min points per file: {np.min(point_counts):,}")
    print(f"Max points per file: {np.max(point_counts):,}")
    print(f"Median points per file: {np.median(point_counts):,.0f}")
    
    print(f"\nFiles with RGB: {sum(has_rgb_list)}/{len(has_rgb_list)}")
    print(f"Files with Intensity: {sum(has_intensity_list)}/{len(has_intensity_list)}")
    
    # Class distribution
    if len(all_classes) > 0:
        unique_classes = sorted(list(set(all_classes)))
        class_counts = Counter(all_classes)
        
        print(f"\nCLASS DISTRIBUTION:")
        print("-" * 80)
        print(f"Total unique classes: {len(unique_classes)}")
        print(f"Classes found: {unique_classes}")
        print()
        print(f"{'Class':<10} {'Occurrences (files)':<25} {'Percentage':<15}")
        print("-" * 80)
        
        for cls in unique_classes:
            count = class_counts[cls]
            percentage = (count / len(all_classes)) * 100
            print(f"{cls:<10} {count:<25} {percentage:<15.2f}%")
    else:
        print("\nNo classification information found in the files.")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDED TRAINING PARAMETERS:")
    print("=" * 80)
    
    # Determine input channels
    input_channels = 3  # XYZ
    if all(has_intensity_list):
        input_channels += 1
        print(f"Use intensity: --use_intensity (all files have intensity)")
    elif any(has_intensity_list):
        print(f"Intensity available in some files (optional: --use_intensity)")
    
    if all(has_rgb_list):
        print(f"Use RGB: --use_rgb (all files have RGB)")
        print(f"Input channels with RGB: {input_channels + 3}")
    elif any(has_rgb_list):
        print(f"RGB available in some files (optional: --use_rgb)")
    
    print(f"\nRecommended input_channels: {input_channels}")
    
    # Number of points recommendation
    avg_points = int(np.mean(point_counts))
    if avg_points < 4096:
        recommended_points = min(2048, avg_points)
    elif avg_points < 10000:
        recommended_points = 4096
    else:
        recommended_points = 8192
    
    print(f"Recommended num_points: {recommended_points}")
    print(f"  (Based on average points per file: {avg_points:,})")
    
    # Batch size recommendation
    if recommended_points <= 2048:
        recommended_batch = 16
    elif recommended_points <= 4096:
        recommended_batch = 8
    else:
        recommended_batch = 4
    
    print(f"Recommended batch_size: {recommended_batch} (adjust based on GPU memory)")
    
    # Number of classes
    if len(unique_classes) > 0:
        # Use max class + 1 to account for 0-indexing
        num_classes = len(unique_classes)
        print(f"Recommended num_classes: {num_classes}")
        print(f"  (Max class ID: {max(unique_classes)})")
    else:
        print(f"No classification found - cannot determine num_classes")
    
    # Example training command
    print("\n" + "=" * 80)
    print("EXAMPLE TRAINING COMMAND:")
    print("=" * 80)
    
    cmd = f"python src/utils/train.py \\\n"
    cmd += f"    --data_dir {data_dir} \\\n"
    cmd += f"    --num_points {recommended_points} \\\n"
    cmd += f"    --batch_size {recommended_batch} \\\n"
    
    if len(unique_classes) > 0:
        cmd += f"    --num_classes {num_classes} \\\n"
    
    if all(has_intensity_list):
        cmd += f"    --use_intensity \\\n"
    
    if all(has_rgb_list):
        cmd += f"    --use_rgb \\\n"
    
    cmd += f"    --epochs 100 \\\n"
    cmd += f"    --lr 0.001 \\\n"
    cmd += f"    --use_scheduler \\\n"
    cmd += f"    --step_size 20 \\\n"
    cmd += f"    --gamma 0.5"
    
    print(cmd)
    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze LAS dataset and get training recommendations')
    parser.add_argument('--data_dir', type=str, default='data_red',
                        help='Path to directory containing .las files')
    args = parser.parse_args()
    
    analyze_las_dataset(args.data_dir)


if __name__ == '__main__':
    main()
