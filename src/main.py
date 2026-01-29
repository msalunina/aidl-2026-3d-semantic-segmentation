if __name__ == '__main__':
    pass

    # TODO: initiate logging

    # TODO: define parameters (both options to read from the command line and default values)
    # we could put a config.yaml file to store default values
    # e.g.:
    # parser = argparse.ArgumentParser(description='Train PointNet for 3D Semantic Segmentation')
    
    # # Data parameters
    # parser.add_argument('--data_dir', type=str, default='data',
    #                     help='Path to data directory')

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

