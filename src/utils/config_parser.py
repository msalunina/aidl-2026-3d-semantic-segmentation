"""
Configuration Parser Module
"""

import yaml


class ConfigParser():
    def __init__(self, default_config_path="config/default.yaml", parser=None):
        self.config_path = default_config_path
        if parser is None:
            raise ValueError("Parser argument is required. Please provide an ArgumentParser instance.")
        self.parser = parser

    def load(self):
        with open(self.config_path, 'r') as file:
            default_config = yaml.safe_load(file)

        # ---- PATHS ---- 
        self.parser.add_argument(
            '--raw_data_path', 
            type=str, 
            default=default_config['paths']['raw_data'], 
            help='Path to raw data directory'
        )
        self.parser.add_argument(
            '--model_data_path', 
            type=str, 
            default=default_config['paths']['model_data'], 
            help='Path to model data directory'
        )
        self.parser.add_argument(
            '--logs_path', 
            type=str, 
            default=default_config['paths']['logs'], 
            help='Path to logs directory'
        )
        self.parser.add_argument(
            '--figures_path', 
            type=str, 
            default=default_config['paths']['figures'], 
            help='Path to figures directory'
        )
        self.parser.add_argument(
            '--models_path', 
            type=str, 
            default=default_config['paths']['models'], 
            help='Path to saved models directory'
        )

        # ---- DATA PREPROCESSING ---- 
        self.parser.add_argument(
            '--block_size', 
            type=float, 
            default=default_config['data_preprocessing']['block_size'], 
            help='Block size in meters'
        )
        self.parser.add_argument(
            '--stride', 
            type=float, 
            default=default_config['data_preprocessing']['stride'], 
            help='Stride in meters'
        )
        self.parser.add_argument(
            '--preprocess_num_points', 
            type=int, 
            default=default_config['data_preprocessing']['num_points'], 
            help='Number of points per block'
        )
        self.parser.add_argument(
            '--min_points_in_block', 
            type=int, 
            default=default_config['data_preprocessing']['min_points_in_block'], 
            help='Minimum number of points in a block'
        )
        self.parser.add_argument(
            '--max_blocks_per_tile', 
            type=lambda x: None if x.lower() == 'none' else int(x), 
            default=default_config['data_preprocessing'].get('max_blocks_per_tile'), 
            help='Maximum number of blocks per tile (use "none" for unlimited)'
        )

        # ---- MODEL ---- 
        self.parser.add_argument(
            '--model_name', 
            type=str, 
            default=default_config['model']['model_name'], 
            help='Model name to use (options: pointnet, TBD)',
            choices=['pointnet']  # TODO: add more options
        )
        self.parser.add_argument(
            '--num_classes', 
            type=int, 
            default=default_config['model']['num_classes'], 
            help='Number of output classes'  # TODO: define classes
        )
        self.parser.add_argument(
            '--num_channels', 
            type=int, 
            default=default_config['model']['num_channels'], 
            help='Number of input channels'  # TODO: define channels
        )

        # ---- TRAINING ---- 
        self.parser.add_argument(
            '--train_num_points', 
            type=int, 
            default=default_config['training']['num_points'], 
            help='Number of points per sample'
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
            choices=['adam']  # TODO: add more options
        )

        # ---- TRAIN TEST SPLIT ---- 
        # self.parser.add_argument(
        #     '--train_ratio', 
        #     type=float, 
        #     default=default_config['train_test_split']['train_ratio'], 
        #     help='Training set ratio'
        # )
        # self.parser.add_argument(
        #     '--test_ratio', 
        #     type=float, 
        #     default=default_config['train_test_split']['test_ratio'], 
        #     help='Test set ratio'
        # )

        self.config = self.parser.parse_args()
        return self.config

    def display(self):

        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)

        for arg in vars(self.config):
            print(f"{arg}: {getattr(self.config, arg)}")
