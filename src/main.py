import argparse
from utils.config_parser import ConfigParser

if __name__ == '__main__':

    # TODO: initiate logging

    config_parser = ConfigParser(
        default_config_path="config/default.yaml",
        parser=argparse.ArgumentParser(description='3D Semantic Segmentation on DALES Dataset')
    )
    config = config_parser.load()
    config_parser.display()

    # print(config.batch_size)


    # TODO: set device

    # TODO: load/split data

    # TODO: initialize model dependent on the provided input
    # e.g.:
    # if args.model == 'pointnet':
    #     from models.pointnet import PointNetSegmentation
    #     model = PointNetSegmentation(...)
    # elif args.model == '...':
    #     ...

    # TODO: define loss function and optimizer (should be the same?)

    # TODO: we could put a learning rate scheduler here

    # TODO: training loop, we could put it into a separate class (trainer.py)

    # TODO: evaluation loop, we could put it into a separate class (evaluator.py) (measure metrics on test set)

    # TODO: save logs to log file

    # ? if we want to compare metrics across different runs, we could save them to a CSV file
    # to display on the same plot after (in a separate script)

