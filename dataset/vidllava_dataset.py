import torch
import os
import av
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

class Videodataset(Dataset):
    def __init__(self, norm, data_dir="/home/dataset/gy/dynamic/"):
        self.data_path = data_dir
        self.objs = os.listdir(data_dir)
        self.videos = []
        self.norm = norm
        
        for root, dirs, files in os.walk(data_dir):
            if ('norm' in root.split("/")) == self.norm:
                for file_name in files:
                    if file_name.lower().endswith(('.mp4')):
                        file_path = os.path.join(root, file_name)
                        self.videos.append(file_path)
        
    def __getitem__(self, idx):
        video_path = self.videos[idx]
        # video = av.open(video_path)
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        clip = read_video_pyav(container, indices)
        
        return clip, video_path
    
    def __len__(self):
        return len(self.videos)
    
class Videodataset_sample(Dataset):
    def __init__(self, data_dir, ratio=1):
        self.data_path = data_dir
        self.objs = os.listdir(data_dir)
        self.videos = []
        # self.clstype = clstype
        self.ratio = ratio
        
        for root, dirs, files in os.walk(data_dir):
            if ('test' in root.split("/")) :
                if not dirs:
                    selected_videos = [os.path.join(root, file_name) for file_name in files if file_name.lower().endswith(('.mp4'))]

                    select_count = max(1, int(len(selected_videos) * self.ratio))
                    
                    selected_selected_videos = random.sample(selected_videos, select_count)
                    self.videos.extend(selected_selected_videos)
        
    def __getitem__(self, idx):
        video_path = self.videos[idx]
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        clip = read_video_pyav(container, indices)
        
        return clip, video_path
    
    def __len__(self):
        return len(self.videos)

class Videodataset_sample_flow(Dataset):
    def __init__(self, clstype, ratio=1, data_dir="/home/lc/Desktop/wza/gy/dyna/Video-LLaVA-main/flow"):
        self.data_path = data_dir
        self.videos = os.listdir(os.path.join(data_dir, "data"))
        # self.flows = os.listdir(os.path.join(data_dir, "flows"))

        

        # for root, dirs, files in os.walk(data_dir):
        #     if ('norm' in root.split("/")) == (self.clstype == "norm") or self.clstype == "all":
        #         if not dirs:
        #             selected_videos = [os.path.join(root, file_name) for file_name in files if file_name.lower().endswith(('.mp4'))]

        #             select_count = max(1, int(len(selected_videos) * self.ratio))
                    
        #             selected_selected_videos = random.sample(selected_videos, select_count)
        #             self.videos.extend(selected_selected_videos)
        
    def __getitem__(self, idx):
        name = self.videos[idx]
        v_path = os.path.join(self.data_path, "data", name)
        f_path = os.path.join(self.data_path, "flows", name)
        
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        clip = read_video_pyav(container, indices)
        
        return clip, video_path
    
    def __len__(self):
        return len(self.videos)