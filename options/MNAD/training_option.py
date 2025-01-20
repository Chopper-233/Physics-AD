import argparse

class TrainOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
    def initialize(self):
        parser = argparse.ArgumentParser()
        parser = argparse.ArgumentParser(description="MNAD")
        parser.add_argument('--gpus', nargs='+', default=None, type=str, help='gpus')
        parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
        parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
        parser.add_argument('--epochs', type=int, default=7, help='number of epochs for training')
        parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss   0.1->0.01')
        parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss 0.1->0.01')
        parser.add_argument('--h', type=int, default=256, help='height of input images')
        parser.add_argument('--w', type=int, default=256, help='width of input images')
        parser.add_argument('--c', type=int, default=3, help='channel of input images')
        parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
        parser.add_argument('--method', type=str, default='pred', help='The target task for anoamly detection')
        parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences  5->1')
        parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
        parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
        parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
        parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
        parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
        parser.add_argument('--dataset_type', type=str, default='dyna', help='type of dataset: ped2, avenue, shanghai')
        parser.add_argument('--dataset_path', type=str, default='/home/dataset/gy/flow', help='directory of data')
        parser.add_argument('--obj', type=str, default='hinge', help='object to be detect')


        self.initialized = True
        self.parser = parser
        return parser