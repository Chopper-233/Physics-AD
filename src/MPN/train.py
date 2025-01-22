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
from tqdm import tqdm
from options.MPN.option import TrainOptions
import warnings
warnings.filterwarnings("ignore") 

parser = TrainOptions().initialize()
args = parser.parse_args()

torch.manual_seed(2020)
obj = args.obj
exp_dir = args.obj

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "1"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus[0]

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

train_folder = os.path.join(args.dataset_path, obj, "train", "frames")

# Loading dataset
train_dataset = VideoDataLoader(train_folder, args.dataset_type, transforms.Compose([
             transforms.ToTensor(),           
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1, segs=args.segs, batch_size=args.batch_size)


train_size = len(train_dataset)

train_batch = data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, drop_last=True)


# Model setting
model = convAE(args.c, args.t_length, args.psize, args.fdim[0], args.pdim[0])
model.cuda()

params_encoder =  list(model.encoder.parameters())
params_decoder = list(model.decoder.parameters())
params_proto = list(model.prototype.parameters())
params_output = list(model.ohead.parameters())
# params = list(model.memory.parameters())
params_D =  params_encoder+params_decoder+params_output+params_proto

optimizer_D = torch.optim.Adam(params_D, lr=args.lr_D)



start_epoch = 0
if os.path.exists(args.resume):
  print('Resume model from '+ args.resume)
  ckpt = args.resume
  checkpoint = torch.load(ckpt)
  start_epoch = checkpoint['epoch']
  model.load_state_dict(checkpoint['state_dict'].state_dict())
  optimizer_D.load_state_dict(checkpoint['optimizer_D'])


# if len(args.gpus[0])>1:
#   model = nn.DataParallel(model)

# Report the training process
log_dir = os.path.join('checkpoints/MPN/', exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not args.debug:
  orig_stdout = sys.stdout
  f = open(os.path.join(log_dir, 'log.txt'),'w')
  sys.stdout= f

loss_func_mse = nn.MSELoss(reduction='none')
loss_pix = AverageMeter()
loss_fea = AverageMeter()
loss_dis = AverageMeter()
# Training


model.train()

for epoch in range(start_epoch, args.epochs):
    labels_list = []
    
    pbar = tqdm(total=len(train_batch))
    for j,(imgs) in enumerate(train_batch):
        imgs = Variable(imgs).cuda()
        imgs = imgs.view(args.batch_size,-1,imgs.shape[-2],imgs.shape[-1])
        # out = model.forward(imgs[:,0:12], None, True)
        outputs, _, _, _, fea_loss, _, dis_loss = model.forward(imgs[:,0:12], None, True)
        optimizer_D.zero_grad()
        loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:,12:]))
        fea_loss = fea_loss.mean()
        dis_loss = dis_loss.mean()
        loss_D = args.loss_fra_reconstruct*loss_pixel + args.loss_fea_reconstruct * fea_loss + args.loss_distinguish * dis_loss 
        loss_D.backward(retain_graph=True)
        optimizer_D.step()


        loss_pix.update(args.loss_fra_reconstruct*loss_pixel.item(), 1)
        loss_fea.update(args.loss_fea_reconstruct*fea_loss.item(), 1)
        loss_dis.update(args.loss_distinguish*dis_loss.item(), 1)

        pbar.set_postfix({
                      'Epoch': '{0} {1}'.format(epoch+1, exp_dir),
                      'Lr': '{:.6f}'.format(optimizer_D.param_groups[-1]['lr']),
                      'PRe': '{:.6f}({:.4f})'.format(loss_pixel.item(), loss_pix.avg),
                      'FRe': '{:.6f}({:.4f})'.format(fea_loss.item(), loss_fea.avg),
                      'Dist': '{:.6f}({:.4f})'.format(dis_loss.item(), loss_dis.avg),
                    })
        pbar.update(1)

    print('----------------------------------------')
    print('Epoch:', epoch+1)
    print('Lr: {:.6f}'.format(optimizer_D.param_groups[-1]['lr']))
    print('PRe: {:.6f}({:.4f})'.format(loss_pixel.item(), loss_pix.avg))
    print('FRe: {:.6f}({:.4f})'.format(fea_loss.item(), loss_fea.avg))
    print('Dist: {:.6f}({:.4f})'.format(dis_loss.item(), loss_dis.avg))
    print('----------------------------------------')   

    pbar.close()

    loss_pix.reset()
    loss_fea.reset()
    loss_dis.reset()

    # Save the model
    if epoch%5==0:
      
      # if len(args.gpus[0])>1:
      #   model_save = model.module
      # else:
      model_save = model
        
      state = {
            'epoch': epoch,
            'state_dict': model_save,
            'optimizer_D' : optimizer_D.state_dict(),
          }
      torch.save(state, os.path.join(log_dir, 'model_'+str(epoch)+'.pth'))

    
print('Training is finished')
if not args.debug:
  sys.stdout = orig_stdout
  f.close()
