import argparse

class Options():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
    def initialize(self):
        parser = argparse.ArgumentParser(description="Demo")
        parser.add_argument("--cfg-path", default='src/VideoLLaMA/eval_configs/video_llama_eval_withaudio.yaml', help="path to configuration file.")
        parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
        parser.add_argument("--model_type", type=str, default='vicuna', help="The type of LLM")
        parser.add_argument("--dataset_path", type=str, default='/home/dataset/gy/dynamic')
        parser.add_argument("--obj", type=str, default='hinge')
        parser.add_argument(
            "--options",
            nargs="+",
            help="override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead.",
        )
        return parser
