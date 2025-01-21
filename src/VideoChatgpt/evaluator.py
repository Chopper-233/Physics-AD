import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from options.VideoChatgpt.option import Options

parser = Options()

args = parser.parse_args()

base_path = 'results/VideoChatgpt/txts'
score_pattern = re.compile(r'Q6: ([\d.]+)')

class_folder = args.obj
class_path = os.path.join(base_path, class_folder)

scores = []
labels = []
for defect in os.listdir(class_path):
    file_path = os.path.join(class_path, defect, "score_responses.txt")

    # 读取并解析txt文件
    with open(file_path, 'r') as f:
        for line in f:
            # 提取分数
            score_match = score_pattern.search(line)
            if score_match:
                score = float(score_match.group(1))
                scores.append(score)
                
                # 设置标签
                if 'anomaly_free' in defect:
                    labels.append(0)  # 正常标签
                else:
                    labels.append(1)  # 异常标签

scores = np.array(scores)
labels = np.array(labels)
auc = roc_auc_score(labels, scores)
ap = average_precision_score(labels, scores)
binary_predictions = (scores >= 0.5).astype(int)
accuracy = accuracy_score(labels, binary_predictions)

with open("results/VideoChatgpt/result.txt",'a') as f:
    print(args.obj, f"AUC: {auc}, AP: {ap}, ACC: {accuracy}", file=f)