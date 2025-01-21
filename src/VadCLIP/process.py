import os
import csv
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--obj',type=str,default='hinge')
args=parser.parse_args()

train_pth="/home/dataset/gy/clipfeature/TrainClipFeatrues"
test_pth="/home/dataset/gy/clipfeature/TestClipFeatrues"

train_list = []
test_list = []
obj = args.obj
train_pth = os.path.join(train_pth,obj)
test_pth = os.path.join(test_pth,obj)

for feat in os.listdir(train_pth):
    if "train" in feat:
        train_list.append([os.path.join(train_pth,feat),'A'])
    elif "anomaly_free" in feat:
        train_list.append([os.path.join(train_pth,feat),'A'])
    else:
        train_list.append([os.path.join(train_pth,feat),'B'])
for feat in os.listdir(test_pth):
    if "anomaly_free" in feat:
        test_list.append([os.path.join(test_pth,feat),'A'])
    else:
        test_list.append([os.path.join(test_pth,feat),'B'])

random.shuffle(train_list)
random.shuffle(test_list)

train_list.insert(0,['path','label'])
test_list.insert(0,['path','label'])

with open('src/VadCLIP/list/train_list.csv','w') as train_file:
    writer = csv.writer(train_file)
    writer.writerows(train_list)
    print(f"changed to {obj}")

with open('src/VadCLIP/list/test_list.csv','w') as test_file:
    writer = csv.writer(test_file)
    writer.writerows(test_list)
    print(f"changed to {obj}")