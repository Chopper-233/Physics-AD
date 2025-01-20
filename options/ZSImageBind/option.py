import argparse

class Options():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
    def initialize(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--obj', type=str, default='hinge')
        parser.add_argument('--gpu', type=int, default=0)
        parser.add_argument('--test_path', type=str, default="/home/dataset/gy/flow/dyna/testing/frames")
        
        self.initialized = True
        self.parser = parser
        return parser