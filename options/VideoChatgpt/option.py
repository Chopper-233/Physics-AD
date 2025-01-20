import argparse

class Options():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
    def initialize(self):
        parser = argparse.ArgumentParser(description="Demo")
        parser.add_argument("--host", type=str, default="0.0.0.0")
        parser.add_argument("--port", type=int)
        parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
        parser.add_argument("--concurrency-count", type=int, default=8)
        parser.add_argument("--model-list-mode", type=str, default="once", choices=["once", "reload"])
        parser.add_argument("--share", action="store_true")
        parser.add_argument("--moderate", action="store_true")
        parser.add_argument("--embed", action="store_true")
        parser.add_argument("--model-name", type=str, default="pretrained-weights/LLaVA-7B-Lightening-v1-1")
        parser.add_argument("--vision_tower_name", type=str, default="openai/clip-vit-large-patch14")
        parser.add_argument("--conv-mode", type=str, default="video-chatgpt_v1")
        parser.add_argument("--obj", type=str, default="hinge")
        parser.add_argument("--data_dir", type=str, default="/home/dataset/gy/dynamic")
        parser.add_argument("--output_dir", type=str, default="results/VideoChatgpt/txts")
        parser.add_argument("--projection_path", type=str, required=False, default="pretrained-weights/Video-ChatGPT-7B/video_chatgpt-7B.bin")
    return parser