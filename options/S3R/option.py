# import argparse
from pathlib import Path

from tap import Tap
from typing import Dict, List, Optional, Tuple, Union

try:
    from typing import Literal # typing.Literal is only available from Python 3.8 and up
except ImportError:
    from typing_extensions import Literal

class S3RArgumentParser(Tap):
    # =============== 
    # network setting
    # ---------------
    backbone: Literal['i3d', 'c3d'] = 'i3d' # default backbone
    feature_size: int = 2048 # size of feature (default: 2048)
    gpus: int = 1
    lr: float = 0.001 # learning rates for steps (list form)
    batch_size: int = 4 # number of instances in a batch of data (default: 32)
    workers: int = 0 # number of workers in dataloader
    model_name: str = 's3r' # name to save model
    obj: Literal['ball', 
            'fan', 
            'rolling_bearing', 
            'spherical_bearing', 
            'servo', 
            'clip', 
            'usb', 
            'hinge', 
            'screw', 
            'lock', 
            'gear', 
            'clock', 
            'slide', 
            'zipper', 
            'button', 
            'rubber_band', 
            'liquid', 
            'caster_wheel', 
            'sticky_roller', 
            'magnet', 
            'toothpaste', 
            'car'] # dataset to train
    plot_freq: int = 10 # frequency of plotting (default: 10)
    max_epoch: int = 1000 # maximum iteration to train (default: 15000)
    dropout: float = 0.7 # dropout ratio
    quantize_size: int = 32 # new temporal size for training

    # ============ 
    # path setting
    # ------------
    root_path: Path = '/home/dataset/gy/S3R/data' # Directory path of data
    log_path: Path = 'src/S3R/logs' # Directory path of log
    checkpoint_path: Path = 'checkpoints/S3R/' # Directory path of log
    dictionary_path: Path ='/home/dataset/gy/S3R/dictionary' # Directory path of dictionary
    resume: Optional[str] = None # trained checkpoint path

    # ========== 
    # evaluation
    # ----------
    evaluate_freq: int = 10 # frequency of running evaluation (default: 1)
    evaluate_min_step: int = 500 # frequency of running evaluation (default: 5000)

    # ==== 
    # misc
    # ----
    seed: Optional[int] = -1 # random seed
    version: str = 'vad-1.0' # experiment version
    debug: bool = False # debug mode
    inference: bool = False # infernece mode
    report_k: int = 10 # maximum reported scores
    descr: List[str] = ['S3R', 'video', 'anomaly', 'detection'] # version description
