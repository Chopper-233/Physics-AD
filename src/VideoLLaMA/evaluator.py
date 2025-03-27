import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from options.VideoLLaMA.option import Options


parser = Options().initialize()

args = parser.parse_args()

# 设置根目录路径
base_path = 'results/VideoLLaMA/txts'

# 正则表达式匹配分数
score_pattern = re.compile(r'{anomalyscore=([\d.]+)}')

# 初始化保存指标的字典
metrics_data = []

# 遍历每个类别文件夹
class_folder = args.obj
class_path = os.path.join(base_path, class_folder)

# 确保是文件夹
# if not os.path.isdir(class_path):
#     continue

# 初始化该类别的score和label列表
scores = []
labels = []

# 遍历该类别文件夹下的每个txt文件
for txt_file in os.listdir(class_path):
    file_path = os.path.join(class_path, txt_file)
    if '_all_' in file_path:
        print(file_path)
        # 读取并解析txt文件
        with open(file_path, 'r') as f:
            for line in f:
                # 提取分数
                score_match = score_pattern.search(line)
                # print(score_match)
                if score_match:
                    score = float(score_match.group(1))
                    scores.append(score)
                    
                    # 设置标签
                    if 'anomaly_free' in txt_file:
                        labels.append(0)  # 正常标签
                    else:
                        labels.append(1)  # 异常标签

        # 保存该类别的score和label到csv文件
        # class_scores_labels = pd.DataFrame({'score': scores, 'label': labels})
        # class_scores_labels.to_csv(f"{base_path}/{class_folder}_scores_labels.csv", index=False)
        
# if len(scores)>0:
# 计算指标
scores = np.array(scores)
labels = np.array(labels)
auc = roc_auc_score(labels, scores)
ap = average_precision_score(labels, scores)
binary_predictions = (scores >= 0.5).astype(int)
accuracy = accuracy_score(labels, binary_predictions)

with open("results/VideoLLaMA/result.txt",'a') as f:
    print(args.obj, f"AUC: {auc}, AP: {ap}, ACC: {accuracy}", file=f)

# # 将指标添加到汇总数据中
# metrics_data.append({
#     'class': class_folder,
#     'accuracy': accuracy,
#     'AUC': auc,
#     'AP': ap
# })

# # if len(metrics_data)>0:
# # 创建指标DataFrame
# metrics_df = pd.DataFrame(metrics_data)

# # 计算所有类别的平均值
# average_metrics = metrics_df.mean(numeric_only=True)
# average_metrics['class'] = 'Average'  # 为平均值行添加类名

# # 将平均值行添加到metrics DataFrame中
# metrics_df = pd.concat([metrics_df, pd.DataFrame([average_metrics])], ignore_index=True)

# # 保存所有类别的指标到一个csv文件
# metrics_df.to_csv(f"{base_path}/metrics.csv", index=False)

