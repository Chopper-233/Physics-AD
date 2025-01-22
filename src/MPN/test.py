import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from dataset.mpn_dataset import *
from models.MPN.base_model import *
from sklearn.metrics import roc_auc_score
from utils.mpn_utils import *
import random
import glob
from tqdm import tqdm
from options.MPN.option import TestOptions
import pdb
import warnings
import time
from sklearn.metrics import auc, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, accuracy_score
warnings.filterwarnings("ignore") 

parser = TestOptions().initialize()
args = parser.parse_args()

torch.manual_seed(2020)

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# if args.gpus is None:
#     gpus = "1"
#     os.environ["CUDA_VISIBLE_DEVICES"]= gpus
# else:
#     gpus = ""
#     for i in range(len(args.gpus)):
#         gpus = gpus + args.gpus[i] + ","
#     os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

obj=args.obj
torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

test_folder = os.path.join(args.dataset_path, obj, "test", "frames")


model_dir = os.path.join(args.model_dir,obj)
# Loading dataset
test_dataset = DynaDataset(test_folder, transforms.Compose([
             transforms.ToTensor(),            
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

test_size = len(test_dataset)

test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

loss_func_mse = nn.MSELoss(reduction='none')


model = convAE(args.c, args.t_length, args.psize, args.fdim[0], args.pdim[0])
model.cuda()

dataset_type = args.dataset_type if args.dataset_type != 'SHTech' else 'shanghai'

# labels = np.load('./data/frame_labels_'+dataset_type+'.npy')
# if 'SHTech' in args.dataset_type or 'ped1' in args.dataset_type:
#     labels = np.expand_dims(labels, 0)

videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
videos_list = [item for item in videos_list if obj == item.split("_")[1]]

gts = []
for i in range(len(videos_list)):
    video_name = videos_list[i].split('/')[-1]
    videos[video_name] = {}
    videos[video_name]['path'] = videos_list[i]
    videos[video_name]['frame'] = glob.glob(os.path.join(videos_list[i], '*.png'))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])
    gts.append(0 if 'anomaly_free' in videos_list[min(i+4, len(videos_list)-1)] else 1)
# print(videos_list)
labels_list = []
anomaly_score_total_list = []
anomaly_score_ae_list = []
anomaly_score_mem_list = []
label_length = 0
psnr_list = {}
feature_distance_list = {}


print('Evaluation of Version {0} on {1}'.format(model_dir.split('/')[-1], args.dataset_type))
# if 'ucf' in model_dir:
#     snapshot_dir = model_dir.replace(args.dataset_type,'UCF')
# else:
#     snapshot_dir = model_dir.replace(args.dataset_type,'SHTech')
snapshot_path = model_dir
psnr_dir = model_dir.replace('exp','results')

# Setting for video anomaly detection
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    if obj == video.split("_")[1]:
        # videos[video_name]['labels'] = labels[0][4+label_length:videos[video_name]['length']+label_length]
        # labels_list = np.append(labels_list, labels[0][args.t_length+args.K_hots-1+label_length:videos[video_name]['length']+label_length])
        label_length += videos[video_name]['length']
        psnr_list[video_name] = []
        feature_distance_list[video_name] = []


if not os.path.isdir(psnr_dir):
    os.makedirs(psnr_dir, exist_ok=True)

ckpt = snapshot_path
ckpt_name = ckpt.split('_')[-1]
# ckpt_id = int(ckpt.split('/')[-1].split('_')[-1][:-4])
# Loading the trained model
model = torch.load(os.path.join(ckpt, args.model_choice))
if type(model) is dict:
    model = model['state_dict']
model.cuda()
model.eval()

# Setting for video anomaly detection
forward_time = AverageMeter()
video_num = 0
update_weights = None
imgs_k = []
k_iter = 0
anomaly_score_total_list.clear()
anomaly_score_ae_list.clear()
anomaly_score_mem_list.clear()
for video in sorted(videos_list):
    if obj == video.split("_")[1]:
        video_name = video.split('/')[-1]
        psnr_list[video_name].clear()
        feature_distance_list[video_name].clear()
pbar = tqdm(total=len(test_batch),
            bar_format='{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}<{remaining}]',)
with torch.no_grad():
    for k,(imgs, label) in enumerate(test_batch):
        hidden_state = None
        imgs = Variable(imgs).cuda()
        
        
        start_t = time.time()
        # out = model.forward(imgs[:,:3*4], update_weights, False)
        outputs, fea_loss = model.forward(imgs[:,:3*4], update_weights, False)
        end_t = time.time()
        
        if k>=len(test_batch)//2:
            forward_time.update(end_t-start_t, 1)
        # import pdb;pdb.set_trace()
        # outputs = torch.cat(pred,1)
        mse_imgs = loss_func_mse((outputs[:]+1)/2, (imgs[:,-3:]+1)/2)

        mse_feas = fea_loss.mean(-1)
        
        mse_feas = mse_feas.reshape((-1,1,256,256))
        mse_imgs = mse_imgs.view((mse_imgs.shape[0],-1))
        mse_imgs = mse_imgs.mean(-1)
        mse_feas = mse_feas.view((mse_feas.shape[0],-1))
        mse_feas = mse_feas.mean(-1)
        # import pdb;pdb.set_trace()
        vid = video_num
        vdd = video_num
        for j in range(len(mse_imgs)):
            psnr_score = psnr(mse_imgs[j].item())
            fea_score = psnr(mse_feas[j].item())
            psnr_list[videos_list[vdd].split('/')[-1]].append(psnr_score)
            feature_distance_list[videos_list[vdd].split('/')[-1]].append(fea_score)
            k_iter += 1
            # print(k_iter, videos[videos_list[video_num].split('/')[-1]]['length']-args.t_length+1)
            if k_iter == videos[videos_list[video_num].split('/')[-1]]['length']-args.t_length+1:
                # gts.append(label[0])
                video_num += 1
                # vdd = video_num
                update_weights = None
                k_iter = 0
                imgs_k = []
                hidden_state = None
            
        pbar.set_postfix({
                        'Epoch': '{0}'.format(ckpt_name),
                        'Vid': '{0}'.format(args.dataset_type+'_'+videos_list[vid].split('/')[-1]),
                        'AEScore': '{:.6f}'.format(psnr_score),
                        'MEScore': '{:.6f}'.format(fea_score),
                        'time': '{:.6f}({:.6f})'.format(end_t-start_t,forward_time.avg),
                        })
        pbar.update(1)

pbar.close()
forward_time.reset()
# print(len(psnr_list['3375_hinge_anomaly_free']))
# Measuring the abnormality score and the AUC
for video in sorted(videos_list):
    if obj == video.split("_")[1]:
        video_name = video.split('/')[-1]
        template = calc(15, 2)
        assert(len(psnr_list)>0 and len(feature_distance_list)>0)
        aa = filter(anomaly_score_list_inv(psnr_list[video_name]), template, 15)
        bb = filter(anomaly_score_list(feature_distance_list[video_name]), template, 15)
        K=5
        topk_scores, _ = torch.topk(torch.tensor(score_sum(aa, bb, args.alpha)), K, dim=0)
        video_score = torch.mean(topk_scores)
        anomaly_score_total_list.append(video_score)

acc_test_list = []
for i in anomaly_score_total_list:
    acc_test_list.append(0 if i<0.5 else 1)

auc = roc_auc_score(gts,anomaly_score_total_list)
ap = average_precision_score(gts, anomaly_score_total_list)
acc = accuracy_score(gts, acc_test_list)

with open("results/MPN/result.txt","a")as f:
    print(obj, "AUC:{:.3}, AP:{:.3f}, ACC:{:.3f}".format(auc, ap, acc), file=f)

# anomaly_score_total = np.asarray(anomaly_score_total_list)
# accuracy_total = 100*AUC(anomaly_score_total, np.expand_dims(1-labels_list, 0))

# print('The result of Version {0} Epoch {1} on {2}'.format(psnr_dir.split('/')[-1], ckpt_name, args.dataset_type))
# print('Total AUC: {:.4f}%'.format(accuracy_total))



