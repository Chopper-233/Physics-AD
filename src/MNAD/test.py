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
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from dataset.mnad_dataset import *
from models.MNAD.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from models.MNAD.Reconstruction import *
from sklearn.metrics import roc_auc_score
from utils.mnad_utils import *
import random
import glob
from sklearn.metrics import auc, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, accuracy_score
from options.MNAD.testing_option import TestOptions
from tqdm import tqdm

parser = TestOptions()
args = parser.parse_args()
obj = args.obj
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

test_folder = args.dataset_path+args.dataset_type+"/testing/frames"

# Loading dataset
test_dataset = DynaDataset(test_folder, transforms.Compose([
             transforms.ToTensor(),            
             ]), resize_height=args.h, resize_width=args.w, obj = obj, time_step=args.t_length-1)

test_size = len(test_dataset)

test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

loss_func_mse = nn.MSELoss(reduction='none')

# Loading the trained model
model = torch.load(args.model_dir)
model.cuda()
m_items = torch.load(args.m_items_dir)
# labels = np.load('./data/frame_labels_'+args.dataset_type+'.npy')

videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
videos_list = [item for item in videos_list if obj == item.split("_")[1]]
# print(len(videos_list))
gts = []
for i in range(len(videos_list)):
    video_name = videos_list[i].split('/')[-1]
    videos[video_name] = {}
    videos[video_name]['path'] = videos_list[i]
    videos[video_name]['frame'] = glob.glob(os.path.join(videos_list[i], '*.png'))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])
    if args.method == 'pred':
        gts.append(0 if 'anomaly_free' in videos_list[min(i+4, len(videos_list)-1)] else 1)
    else:
        gts.append(0 if 'anomaly_free' in videos_list[i] else 1)

# print(len(gts))
labels_list = []
anomaly_score_total_list = []
anomaly_score_ae_list = []
anomaly_score_mem_list = []
label_length = 0
psnr_list = {}
feature_distance_list = {}

print('Evaluation of', args.dataset_type)

# Setting for video anomaly detection
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    if obj == video.split("_")[1]:
        # if args.method == 'pred':
        #     labels_list = np.append(labels_list, labels[0][4+label_length:videos[video_name]['length']+label_length])
        # else:
        #     labels_list = np.append(labels_list, labels[0][label_length:videos[video_name]['length']+label_length])
        label_length += videos[video_name]['length']
        psnr_list[video_name] = []
        feature_distance_list[video_name] = []

label_length = 0
video_num = 0
label_length += videos[videos_list[video_num].split('/')[-1]]['length']
m_items_test = m_items.clone()

model.eval()

for k,(imgs) in enumerate(tqdm(test_batch)):
    
    if args.method == 'pred':
        if k == label_length-4*(video_num+1):
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    else:
        if k == label_length:
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    imgs = Variable(imgs).cuda()
    
    if args.method == 'pred':
        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
        mse_feas = compactness_loss.item()

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs[:,3*4:])
    
    else:
        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(imgs, m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0]+1)/2)).item()
        mse_feas = compactness_loss.item()

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs)

    if  point_sc < args.th:
        query = F.normalize(feas, dim=1)
        query = query.permute(0,2,3,1) # b X h X w X d
        m_items_test = model.memory.update(query, m_items_test, False)

    psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))
    feature_distance_list[videos_list[video_num].split('/')[-1]].append(mse_feas)
# print(feature_distance_list)

# Measuring the abnormality score and the AUC
anomaly_score_total_list = []
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    video_score = score_sum(anomaly_score_list_inv(psnr_list[video_name]), 
                                     anomaly_score_list(feature_distance_list[video_name]), args.alpha)
    # print(score_sum(anomaly_score_list(psnr_list[video_name]), 
    #                                  anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha))
    K=5
    topk_scores, _ = torch.topk(torch.tensor(video_score), K, dim=0)
    video_score = torch.mean(topk_scores)
    anomaly_score_total_list.append(video_score)
# print(len(anomaly_score_total_list))
anomaly_score_total_list = np.asarray(anomaly_score_total_list)

acc_test_list = []
for i in anomaly_score_total_list:
    acc_test_list.append(0 if i<0.5 else 1)

auc = roc_auc_score(gts,anomaly_score_total_list)
ap = average_precision_score(gts, anomaly_score_total_list)
acc = accuracy_score(gts, acc_test_list)
# accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))


with open("results/MNAD/result.txt","a")as f:
    print(obj, "AUC:{:.3}, AP:{:.3f}, ACC:{:.3f}".format(auc, ap, acc), file=f)
# print('The result of ', args.dataset_type)
# print('AUC: ', accuracy*100, '%')
