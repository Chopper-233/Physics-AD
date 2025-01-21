import argparse

class TrainOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
    def initialize(self):
        parser = argparse.ArgumentParser(description="MPN")
        parser.add_argument('--gpus', nargs='+', default=None, type=str, help='gpus')
        parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
        parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
        parser.add_argument('--epochs', type=int, default=21, help='number of epochs for training')
        parser.add_argument('--loss_fra_reconstruct', type=float, default=1.00, help='weight of the frame reconstruction loss')
        parser.add_argument('--loss_fea_reconstruct', type=float, default=1.00, help='weight of the feature reconstruction loss')
        parser.add_argument('--loss_distinguish', type=float, default=0.0001, help='weight of the feature distinction loss')
        parser.add_argument('--h', type=int, default=256, help='height of input images')
        parser.add_argument('--w', type=int, default=256, help='width of input images')
        parser.add_argument('--c', type=int, default=3, help='channel of input images')
        parser.add_argument('--lr_D', type=float, default=1e-4, help='initial learning rate for parameters')
        parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
        parser.add_argument('--segs', type=int, default=32, help='num of video segments')
        parser.add_argument('--fdim', type=list, default=[128], help='channel dimension of the features')
        parser.add_argument('--pdim', type=list, default=[128], help='channel dimension of the prototypes')
        parser.add_argument('--psize', type=int, default=10, help='number of the prototype items')
        parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
        parser.add_argument('--num_workers', type=int, default=8, help='number of workers for the train loader')
        parser.add_argument('--num_workers_test', type=int, default=8, help='number of workers for the test loader')
        parser.add_argument('--dataset_type', type=str, default='phys-ad', help='type of dataset')
        parser.add_argument('--dataset_path', type=str, default='/home/dataset/gy/phys/', help='directory of data')
        parser.add_argument('--obj', default='ball', type=str, help='object to detect')
        parser.add_argument('--resume', type=str, default='checkpoints/MPN/pretrain_model.pth', help='file path of resume pth')
        parser.add_argument('--debug', type=bool, default=False, help='if debug')


        self.initialized = True
        self.parser = parser
        return parser

class TestOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
    def initialize(self):
        parser = argparse.ArgumentParser(description="MPN")
        parser.add_argument('--gpu', default='0', type=str, help='gpus')
        parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
        parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
        parser.add_argument('--h', type=int, default=256, help='height of input images')
        parser.add_argument('--w', type=int, default=256, help='width of input images')
        parser.add_argument('--c', type=int, default=3, help='channel of input images')
        parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
        parser.add_argument('--fdim', type=list, default=[128], help='channel dimension of the features')
        parser.add_argument('--pdim', type=list, default=[128], help='channel dimension of the prototypes')
        parser.add_argument('--psize', type=int, default=10, help='number of the prototypes')
        parser.add_argument('--test_iter', type=int, default=1, help='channel of input images')
        parser.add_argument('--K_hots', type=int, default=1, help='number of the K hots')
        parser.add_argument('--alpha', type=float, default=0.5, help='weight for the anomality score')
        parser.add_argument('--th', type=float, default=0.0, help='threshold for test updating')
        parser.add_argument('--num_workers_test', type=int, default=8, help='number of workers for the test loader')
        parser.add_argument('--dataset_type', type=str, default='phys-ad', help='type of dataset')
        parser.add_argument('--dataset_path', type=str, default='/home/dataset/gy/phys/', help='directory of data')
        parser.add_argument('--model_dir', default="checkpoints/MPN/", type=str, help='directory of model')
        parser.add_argument('--obj',default="ball", type=str, help='object to detect')
        parser.add_argument('--model_choice',default="model_20.pth", type=str, help='model of different epoches')

        self.initialized = True
        self.parser = parser
        return parser
    
