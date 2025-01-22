import os
import models.ZSCLIP as clip
import torch
from torchvision.datasets import CIFAR100
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score
from tqdm import tqdm
from options.ZSCLIP.option import Options


parser = Options().initialize()
args = parser.parse_args()
obj = args.obj
train_feature = args.train_feat_path
test_feature = args.test_feat_path
device = args.gpu if torch.cuda.is_available() else "cpu"

# Load the model
model, preprocess = clip.load(args.clip_model, device)
# objs = os.listdir(test_feature)

with open("results/ZSCLIP/result.txt","a") as f:
    # for obj in tqdm(objs):
        # if obj == "ball":
    pred = []
    gt = []
    text_norm_inputs = clip.tokenize(f"a video of a well functioning {obj}").to(device)
    text_abnorm_inputs = clip.tokenize(f"a video of a {obj} which is not properly functioning" ).to(device)
    # for abn in os.path.join(test_feature, obj):
    # torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
    
    for test_vid in os.listdir(os.path.join(test_feature,obj)):
        image_features = torch.from_numpy(np.load(os.path.join(test_feature,obj,test_vid))).to(device).to(torch.float16)
        abn = test_vid.split("_")[-3:-1]
        gt.append(0 if "_".join(abn) == "anomaly_free" else 1)
        # Calculate features
        with torch.no_grad():
            # image_features = model.encode_image(image_input)
            # text_norm_features = model.encode_text(text_norm_inputs)
            text_abnorm_features = model.encode_text(text_abnorm_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # text_norm_features /= text_norm_features.norm(dim=-1, keepdim=True)
        text_abnorm_features /= text_abnorm_features.norm(dim=-1, keepdim=True)

        # similarity_n = (100.0 * image_features @ text_norm_features.T).softmax(dim=-1)
        # values_norm, indices_norm = similarity_n[0].topk(1)

        similarity_abn = (100.0 * image_features @ text_abnorm_features.T).softmax(dim=-1)
        values_abnorm, indices_abnorm = similarity_abn[0].topk(1)

        pred.append(values_abnorm[0].cpu())

        # Print the result
        # print("\nTop predictions:\n")
        # for value, index in zip(values, indices):
        #     print(f"{obj}: {100 * value.item():.2f}%")
    
    AUC = roc_auc_score(gt,pred)
    AP = average_precision_score(gt,pred)
    ACC = accuracy_score(gt,pred)
    print(obj, "AUC: {:.3f}".format(AUC), "AP: {:.3f}".format(AP), "ACC: {:.3f}".format(ACC), file=f)