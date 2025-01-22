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
from sklearn.metrics import roc_auc_score
from utils.mnad_utils import *
import random
from tqdm import tqdm
from options.MNAD.option import TrainOptions

parser = TrainOptions().initialize()
args = parser.parse_args()
obj = args.obj
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "6"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

train_folder = os.path.join(args.dataset_path, obj, "train", "frames")
test_folder = os.path.join(args.dataset_path, obj, "test", "frames")

# Loading dataset
train_dataset = DynaDataset(train_folder, transforms.Compose([
             transforms.ToTensor(),          
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

# test_dataset = DynaDataset(test_folder, transforms.Compose([
#              transforms.ToTensor(),            
#              ]), resize_height=args.h, resize_width=args.w, obj="clip", time_step=args.t_length-1)

train_size = len(train_dataset)
#test_size = len(test_dataset)

train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
# test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
#                              shuffle=False, num_workers=args.num_workers_test, drop_last=False)


# Model setting
assert args.method == 'pred' or args.method == 'recon', 'Wrong task name'
if args.method == 'pred':
    from models.MNAD.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
    model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)
else:
    from models.MNAD.Reconstruction import *
    model = convAE(args.c, memory_size = args.msize, feature_dim = args.fdim, key_dim = args.mdim)
params_encoder =  list(model.encoder.parameters()) 
params_decoder = list(model.decoder.parameters())
params = params_encoder + params_decoder
optimizer = torch.optim.Adam(params, lr = args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)
model.cuda()


# Report the training process
log_dir = os.path.join('checkpoints/MNAD/', args.method, args.obj)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'),'w')
sys.stdout= f

loss_func_mse = nn.MSELoss(reduction='none')

# Training

m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda() # Initialize the memory items

for epoch in range(args.epochs):
    labels_list = []
    model.train()
    
    start = time.time()
    for imgs in tqdm(train_batch):
        # pass
        imgs = Variable(imgs).cuda()
        
        if args.method == 'pred':
            outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(imgs[:,0:12], m_items, True)
        
        else:
            outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(imgs, m_items, True)
        
        
        optimizer.zero_grad()
        if args.method == 'pred':
            loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:,12:]))
        else:
            loss_pixel = torch.mean(loss_func_mse(outputs, imgs))
            
        loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
        loss.backward(retain_graph=True)
        optimizer.step()
        # print("a batch")
    scheduler.step()
    
    print('----------------------------------------')
    print('Epoch:', epoch+1)
    if args.method == 'pred':
        print('Loss: Prediction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(loss_pixel.item(), compactness_loss.item(), separateness_loss.item()))
    else:
        print('Loss: Reconstruction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(loss_pixel.item(), compactness_loss.item(), separateness_loss.item()))
    print('Memory_items:')
    print(m_items)
    print('----------------------------------------')
    
print('Training is finished')
# Save the model and the memory items
torch.save(model, os.path.join(log_dir, 'model.pth'))
torch.save(m_items, os.path.join(log_dir, 'keys.pt'))
    
sys.stdout = orig_stdout
f.close()



