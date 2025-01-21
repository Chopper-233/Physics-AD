''' This config file will handle the video anomaly detection'''

from pathlib import Path

data_root = Path('/home/dataset/gy/S3R/data/')
dictionary_root = Path('/home/dataset/gy/S3R/dictionary/')
quantize_size = 32
backbone = 'i3d'


