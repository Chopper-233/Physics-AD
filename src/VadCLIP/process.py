import os
import csv
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--obj',type=str,default='hinge')
args=parser.parse_args()

train_pth="/home/dataset/gy/vadclipfeature/DynaTrainClipFeatrues"
test_pth="/home/dataset/gy/vadclipfeature/DynaTestClipFeatrues"

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


# csv_file = "/home/lc/Desktop/wza/gy/dyna/bench/VadCLIP-main/list/xd_CLIP_rgbtest.csv"
# count=0
# abn_count=0
# with open(csv_file, 'r', encoding='utf-8') as csvfile:
#     csvreader = csv.reader(csvfile)
    
#     # 跳过表头（如果有表头）
#     next(csvreader)

#     # 逐行读取CSV文件
#     for row in csvreader:
#         count+=1
#         # 将每行转换为字符串进行搜索
#         abntype = row[1]
#         if 'B' in abntype or 'G' in abntype:
#            abn_count+=1

# print(count,abn_count,abn_count/count)
# data = np.load("/home/lc/Desktop/wza/gy/dyna/bench/VadCLIP-main/list/gt_label.npy",allow_pickle=True)
# data = np.load("/home/lc/Desktop/wza/gy/dyna/bench/VadCLIP-main/list/gt_segment.npy",allow_pickle=True)
# datalist = os.listdir("/home/lc/Desktop/wza/gy/dyna/bench/VadCLIP-main/data/DynaTestClipFeatrues")
# print(len(datalist))