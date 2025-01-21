import os
from sklearn.metrics import auc, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, accuracy_score
import numpy as np
result = "results/MemAE/tmp/"
with open("results/MemAE/result.txt","w") as f:
    for obj in os.listdir(result):
        pred = []
        gt = []
        for vid in os.listdir(os.path.join(result, obj)):
            frame_scores = np.load(os.path.join(result, obj, vid))
            # print(vid_score.shape)
            if "anomaly_free" in vid:
                gt.append(0)
            else:
                gt.append(1)
            K = 5
            top_k_indices = np.argsort(frame_scores)[-K:][::-1]
            top_k_values = frame_scores[top_k_indices]
            vid_score = sum(top_k_values)/K
            pred.append(vid_score)
        auc = roc_auc_score(gt, pred)
        ap = average_precision_score(gt, pred)
        pred_acc = [0 if i < 0.5 else 1 for i in pred]
        acc = accuracy_score(gt, pred_acc)
        print(obj, "{:.3f},{:.3f},{:.3f}".format(auc,ap,acc), file=f)