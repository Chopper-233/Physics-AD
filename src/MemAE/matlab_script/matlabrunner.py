import os
import glob
import numpy as np
from scipy.io import savemat

def mkdirfunc(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

data_root_path = '/home/dataset/gy/'
in_path = os.path.join(data_root_path, 'phys')
frame_file_type = 'png'
clip_len = 16  # number of frames in a clip
overlap_rate = 0
skip_step = 1
clip_rng = clip_len * skip_step - 1
overlap_shift = clip_len - 1  # full overlap shift


sub_dir_list = os.listdir(in_path)
tasks = ['train', 'test']

for task in tasks:
    for sub_dir_name in sub_dir_list:
        print(sub_dir_name)
        sub_in_path = os.path.join(in_path, sub_dir_name, task, 'frames')
        idx_out_path = os.path.join(in_path, f'{sub_dir_name}_idx')
        mkdirfunc(idx_out_path)

        v_list = [d for d in os.listdir(sub_in_path) if os.path.isdir(os.path.join(sub_in_path, d))]
        
        for v_name in v_list:
            print(v_name)
            video_path = os.path.join(sub_in_path, v_name)

            frame_list = glob.glob(os.path.join(video_path, f'*.{frame_file_type}'))
            frame_num = len(frame_list)

            s_list = np.arange(1, frame_num - clip_rng + 1, clip_rng + 1 - overlap_shift)
            e_list = s_list + clip_rng

            valid_idx = e_list <= frame_num
            s_list = s_list[valid_idx]
            e_list = e_list[valid_idx]

            video_sub_dir_out_path = os.path.join(idx_out_path, v_name)
            mkdirfunc(video_sub_dir_out_path)

            for j, (s, e) in enumerate(zip(s_list, e_list)):
                idx = np.arange(s, e + 1, skip_step)
                idx_filename = os.path.join(video_sub_dir_out_path, f'{v_name}_i{j + 1:03d}.mat')
                savemat(idx_filename, {'v_name': v_name, 'idx': idx})