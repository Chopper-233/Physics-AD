from __future__ import absolute_import, print_function
import os
from utils import memae_utils as utils
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from dataset import memae_dataset as data
import scipy.io as sio
from options.MemAE.testing_options import TestOptions
import time
from models.MemAE import AutoEncoderCov3D, AutoEncoderCov3DMem
from tqdm import tqdm
###
opt_parser = TestOptions()
opt = opt_parser.parse(is_print=True)
use_cuda = opt.UseCUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if use_cuda else "cpu")
obj=opt.obj
###
batch_size_in = opt.BatchSize #1
chnum_in_ = opt.ImgChnNum      # channel number of the input images
framenum_in_ = opt.FrameNum  # frame number of the input images in a video clip
mem_dim_in = opt.MemDim
sparse_shrink_thres = opt.ShrinkThres

img_crop_size = 0

######
model_setting = utils.get_model_setting(opt)

## data path
data_root = os.path.join(opt.DataRoot, obj, 'test')
tr_data_frame_dir = os.path.join(data_root, 'frames')
tr_data_idx_dir = os.path.join(data_root, 'frames_idx')

############ model path
model_path = opt.ModelFilePath

### test result path
te_res_root = opt.OutRoot
te_res_path = os.path.join(te_res_root, "tmp", obj)
utils.mkdir(te_res_path)

###### loading trained model
if (opt.ModelName == 'AE'):
    model = AutoEncoderCov3D(chnum_in_)
elif(opt.ModelName=='MemAE'):
    model = AutoEncoderCov3DMem(chnum_in_, mem_dim_in, shrink_thres=sparse_shrink_thres)
else:
    model = []
    print('Wrong Name.')

##
model_para = torch.load(model_path)
model.load_state_dict(model_para)
model.to(device)
model.eval()

##
if(chnum_in_==1):
    norm_mean = [0.5]
    norm_std = [0.5]
elif(chnum_in_==3):
    norm_mean = (0.5, 0.5, 0.5)
    norm_std = (0.5, 0.5, 0.5)

frame_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
unorm_trans = utils.UnNormalize(mean=norm_mean, std=norm_std)

# ##
video_list = utils.get_subdir_list(data_idx_dir, obj=obj)
video_num = len(video_list)

##
with torch.no_grad():
    for ite_vid in range(video_num):
        video_name = video_list[ite_vid]
        video_idx_path = os.path.join(data_idx_dir, video_name)  # idx path of the current sub dir
        video_frame_path = os.path.join(data_frame_dir, video_name)  # frame path of the current sub dir
        # info for current video
        idx_name_list = [name for name in os.listdir(video_idx_path) \
                         if os.path.isfile(os.path.join(video_idx_path, name)) and obj == name.split("_")[1]]
        idx_name_list.sort()
        # load data (frame clips) for single video
        video_dataset = data.VideoDatasetOneDir(video_idx_path, video_frame_path, obj=obj, transform=frame_trans)
        video_data_loader = DataLoader(video_dataset,
                                       batch_size=batch_size_in,
                                       num_workers=16,
                                       shuffle=False
                                       )
        # testing results on a single video sequence
        print('[vidx %02d/%d] [vname %s]' % (ite_vid+1, video_num, video_name))
        recon_error_list = []
        #
        for batch_idx, (item, frames) in enumerate(tqdm(video_data_loader)):
            idx_name = idx_name_list[item[0]]
            idx_data = sio.loadmat(os.path.join(video_idx_path, idx_name))
            v_name = idx_data['v_name'][0]  # video name
            frame_idx = idx_data['idx'][0, :]  # frame index list for a video clip
            ######
            frames = frames.to(device)
            ##
            if (opt.ModelName == 'AE'):
                recon_frames = model(frames)
                ###### calculate reconstruction error (MSE)
                recon_np = utils.vframes2imgs(unorm_trans(recon_frames.data), step=1, batch_idx=0)
                input_np = utils.vframes2imgs(unorm_trans(frames.data), step=1, batch_idx=0)
                r = utils.crop_image(recon_np, img_crop_size) - utils.crop_image(input_np, img_crop_size)
                # recon_error = np.mean(sum(r**2)**0.5)
                recon_error = np.mean(r ** 2)  # **0.5
                recon_error_list += [recon_error]
            elif (opt.ModelName == 'MemAE'):
                recon_res = model(frames)
                recon_frames = recon_res['output'].resize_(frames.shape)
                r = recon_frames - frames
                r = utils.crop_image(r, img_crop_size)
                sp_error_map = torch.sum(r**2, dim=1)**0.5
                s = sp_error_map.size()
                sp_error_vec = sp_error_map.view(s[0], -1)
                recon_error = torch.mean(sp_error_vec, dim=-1)
                recon_error_list += recon_error.cpu().tolist()
            ######
            # elif (opt.ModelName == 'MemAE'):
            #     recon_res = model(frames)
            #     recon_frames = recon_res['output']
            #     recon_np = utils.btv2btf(unorm_trans(recon_frames.data))
            #     input_np = utils.btv2btf(unorm_trans(frames.data))
            #     r = utils.crop_image(recon_np, img_crop_size) - utils.crop_image(input_np, img_crop_size)
            #     sp_error_map = np.sum(r**2, axis=1)**0.5
            #     tmp_s = sp_error_map.shape
            #     sp_error_vec = np.reshape(sp_error_map, (tmp_s[0], -1))
            #     recon_error = np.mean(sp_error_vec, axis=-1)
            #     recon_error_list += recon_error.tolist()
            #######
            else:
                recon_error = -1
                print('Wrong ModelName.')
        os.makedirs(os.path.join(te_res_path), exist_ok=True)
        np.save(os.path.join(te_res_path, video_name + '.npy'), recon_error_list)

## evaluation
# utils.eval_video(data_root, te_res_path, is_show=False)
