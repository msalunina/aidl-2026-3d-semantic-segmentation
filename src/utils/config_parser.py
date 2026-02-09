"""
Configuration Parser Module
"""

import yaml
from argparse import Namespace


class ConfigParser():
    def __init__(self, default_config_path="config/default.yaml", parser=None):
        self.config_path = default_config_path
        if parser is None:
            raise ValueError("Parser argument is required. Please provide an ArgumentParser instance.")
        self.parser = parser

    def load(self):
        # Load default config from YAML
        with open(self.config_path, 'r') as file:
            default_config = yaml.safe_load(file)

        # ---- COMMAND LINE OVERRIDABLE PARAMETERS ----
        # All training parameters can be overridden from command line
        self.parser.add_argument(
            '--model_name', 
            type=str, 
            default=default_config['training']['model_name'], 
            help='Model name to use (options: pointnet, TBD)',
            choices=['pointnet']  # TODO: add more options
        )
        self.parser.add_argument(
            '--num_channels', 
            type=int, 
            default=default_config['training']['num_channels'], 
            help='Number of input channels (e.g., 3 for XYZ)'
        )
        self.parser.add_argument(
            '--num_points', 
            type=int, 
            default=default_config['training']['num_points'], 
            help='Number of points per sample during training'
        )
        self.parser.add_argument(
            '--batch_size', 
            type=int, 
            default=default_config['training']['batch_size'], 
            help='Batch size for training'
        )
        self.parser.add_argument(
            '--num_epochs', 
            type=int, 
            default=default_config['training']['num_epochs'], 
            help='Number of training epochs'
        )
        self.parser.add_argument(
            '--learning_rate', 
            type=float,
            default=default_config['training']['learning_rate'], 
            help='Learning rate for optimizer'
        )
        self.parser.add_argument(
            '--dropout_rate', 
            type=float,
            default=default_config['training']['dropout_rate'], 
            help='Dropout rate for model'
        )
        self.parser.add_argument(
            '--optimizer', 
            type=str, 
            default=default_config['training']['optimizer'], 
            help='Optimizer to use for training',
            choices=['adam']  # TODO: add more options as needed
        )

        # Parse command line arguments
        cmd_args = self.parser.parse_args()

        # ---- CREATE FULL CONFIG ----
        # Start with default config and merge CLI arguments
        config_dict = self._flatten_config(default_config)
        
        # Infer num_classes from class_mapping if not explicitly set
        if 'num_classes' not in config_dict and 'class_mapping' in config_dict:
            config_dict['num_classes'] = len(set(config_dict['class_mapping'].values()))
        
        # Convert RGB lists to tuples in 3D visualization color mapping
        if 'viz_3d' in config_dict and 'color_mapping' in config_dict['viz_3d']:
            config_dict['viz_3d']['color_mapping'] = {
                k: tuple(v) if isinstance(v, list) else v 
                for k, v in config_dict['viz_3d']['color_mapping'].items()
            }
        
        # Override with command line arguments (all training parameters)
        config_dict['model_name'] = cmd_args.model_name
        config_dict['num_channels'] = cmd_args.num_channels
        config_dict['train_num_points'] = cmd_args.num_points
        config_dict['batch_size'] = cmd_args.batch_size
        config_dict['num_epochs'] = cmd_args.num_epochs
        config_dict['learning_rate'] = cmd_args.learning_rate
        config_dict['dropout_rate'] = cmd_args.dropout_rate
        config_dict['optimizer'] = cmd_args.optimizer

        # Convert to Namespace for backward compatibility
        self.config = Namespace(**config_dict)
        return self.config

    def _flatten_config(self, config_dict, parent_key='', sep='_'):
        """
        Flatten nested config dictionary into a single-level dict.
        Example: {'paths': {'raw_data': './data'}} -> {'raw_data_path': './data'}
        Special handling for visualization settings to keep sub-dictionaries intact.
        """
        items = []
        for key, value in config_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            # Keep certain nested structures intact
            preserve_nested = key in ['class_mapping', 'color_mapping', '2d', '3d']
            preserve_parent = parent_key == 'visualization'
            
            if isinstance(value, dict) and not preserve_nested and not preserve_parent:
                # Recursively flatten most dicts
                items.extend(self._flatten_config(value, new_key, sep=sep).items())
            else:
                # Rename for consistency with old naming
                if parent_key == 'paths':
                    new_key = f"{key}_path"
                elif parent_key == 'data_preprocessing':
                    if key == 'num_points':
                        new_key = 'preprocess_num_points'
                    else:
                        new_key = key
                elif parent_key == 'training':
                    if key == 'num_points':
                        new_key = 'train_num_points'
                    else:
                        new_key = key
                elif parent_key == 'dataset':
                    # Keep dataset settings with dataset_ prefix
                    new_key = f"dataset_{key}"
                elif parent_key == 'visualization':
                    # Keep visualization settings as viz_2d and viz_3d
                    new_key = f"viz_{key}"
                    
                items.append((new_key, value))
        
        return dict(items)

    def display(self):

        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)

        for arg in vars(self.config):
            print(f"{arg}: {getattr(self.config, arg)}")
