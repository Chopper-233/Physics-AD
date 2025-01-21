from src.ZSImageBind import data
import torch
from models.ZSImageBind import imagebind_model
from models.ZSImageBind.imagebind_model import ModalityType
import os
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, accuracy_score
import numpy as np
from options.ZSImageBind.option import Options

parser = Options()
args = parser.parse_args()

obj = args.obj
testfile = args.test_path

video_paths = {}
for vid in os.listdir(testfile):
    img_paths = []
    for img in os.listdir(os.path.join(testfile, vid)):
        img_paths.append(os.path.join(testfile, vid, img))
    # for obj in obj_list:
    if obj == vid.split("_")[1]:
        if obj not in video_paths.keys():
            video_paths[obj] = [img_paths]
        else:
            video_paths[obj].append(img_paths)

device = args.gpu if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

with open("results/ZSImageBind/result.txt","a") as f:
    gt = []
    pred = []
    for img_paths in tqdm(video_paths[obj]):
        text_list = ['A well functioning '+obj for _ in range(len(img_paths))]
        # Load data
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(text_list, device),
            ModalityType.VISION: data.load_and_transform_vision_data(img_paths, device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        det = torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1)
        # # print(det.size())
        gt.append(0.0 if "anomaly_free" in img_paths[0] else 1.0)
        pred.append("{:.3f}".format(1-torch.linalg.det(det)))
    gt = np.array(gt,dtype=np.float64)
    pred = np.array(pred,dtype=np.float64)

    auc = roc_auc_score(gt,pred)
    ap = average_precision_score(gt,pred)
    pred_acc = [0 if i<0.5 else 1 for i in pred]
    pred_acc = np.array(pred_acc,dtype=np.float64)
    acc = accuracy_score(gt,pred)
    print(obj, "AUC: {:.3f}, AP: {:.3f}, ACC: {:.3f}".format(auc,ap,acc), file=f)

