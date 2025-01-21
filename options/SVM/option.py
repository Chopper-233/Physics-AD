import argparse

class Options():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
    def initialize(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--obj', type=str, default='hinge')
        parser.add_argument('--feat_path', type=str, default="/home/dataset/gy/i3d_feature/")
        
        self.initialized = True
        self.parser = parser
        return parser