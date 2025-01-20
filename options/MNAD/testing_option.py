import argparse

class TestOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
    def initialize(self):
        parser = argparse.ArgumentParser(description="MNAD")
        parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
        parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
        parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
        parser.add_argument('--h', type=int, default=256, help='height of input images')
        parser.add_argument('--w', type=int, default=256, help='width of input images')
        parser.add_argument('--c', type=int, default=3, help='channel of input images')
        parser.add_argument('--method', type=str, default='pred', help='The target task for anoamly detection')
        parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
        parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
        parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
        parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
        parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
        parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
        parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
        parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
        parser.add_argument('--dataset_type', type=str, default='dyna', help='type of dataset: ped2, avenue, shanghai, dyna')
        parser.add_argument('--dataset_path', type=str, default='/home/dataset/gy/flow/', help='directory of data')
        parser.add_argument('--model_dir', type=str, default = 'checkpoints/MNAD/pred/hinge/model.pth', help='directory of model')
        parser.add_argument('--m_items_dir', type=str, default = 'checkpoints/MNAD/pred/hinge/keys.pt', help='directory of model')
        parser.add_argument('--obj',default="hinge", type=str)


        self.initialized = True
        self.parser = parser
        return parser