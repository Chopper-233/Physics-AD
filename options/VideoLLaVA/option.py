import argparse

class Options():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
    def initialize(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--task', type=str, default='auc', choices=["describ","reason","auc"])
        parser.add_argument('--local_rank', type=int)
        parser.add_argument('--model_path', type=str, default="pretrained-weights/Video-LLaVA-7B-hf")
        parser.add_argument('--config', type=str, default="src/VideoLLaVA/ds_inference_config.json")
        parser.add_argument('--data_dir', type=str, default="/home/dataset/gy/dynamic/")
        
        self.initialized = True
        self.parser = parser
        return parser