import os
import json
import heapq
from sklearn.metrics import auc, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, accuracy_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--obj", type=str)
parser.add_argument("--root_path", type=str)
args = parser.parse_args()
obj = args.obj
json_path = f"{args.root_path}/{obj}/scores/refined/llama-2-13b-chat/opt-6.7b-coco/501951_if_you_were_a_law_enforcement_agency,_how_would_you_rate_the_scene_described_on_a_scale_from_0_to_1,_with_0_representing_a_standard_scene_and_1_denoting_a_scene_with_suspicious_activities?"

# obj_list = []
with open("results/LAVAD/result.txt","a") as outfile:
    # for obj in obj_list:
    gt = []
    pred = []
    for file in os.listdir(json_path):
        if obj in file.split("_"):
            if "anomaly_free" in file:
                gt.append(0)
            else:
                gt.append(1)
            with open(os.path.join(json_path, file)) as f:
                data = json.load(f)
                scores_for_vid = []
                for key in data.keys():
                    scores = heapq.nlargest(3, [score for score in data[key].values()])
                    score_for_key_frame = sum(scores)/len(scores)
                    scores_for_vid.append(score_for_key_frame)
                final_score_for_vid = heapq.nlargest(5, scores_for_vid)
                final_score_for_vid = sum(final_score_for_vid)/len(final_score_for_vid)
                pred.append(final_score_for_vid)

    auc = roc_auc_score(gt,pred)
    ap = average_precision_score(gt,pred)

    acc_pred = [0 if i<0.5 else 1 for i in pred ]
    acc = accuracy_score(gt, acc_pred)

    print(obj, "{:.3f},{:.3f},{:.3f}".format(auc, ap, acc), file=outfile)