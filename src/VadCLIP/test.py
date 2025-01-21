import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
import os
from models.VadCLIP.model import CLIPVAD
from dataset.vadclip_dataset import *
from utils.vadclip_utils import get_batch_mask, get_prompt_text
# from utils.dyna_detectionMAP import getDetectionMAP as dmAP
from utils.vadclip_vid_level_pred import getvidpred
from options.VadCLIP import option
from tqdm import tqdm

def test(model, testdataloader, maxlen, prompt_text, device, gt=None, gtsegments=None, gtlabels=None):
    
    model.to(device)
    model.eval()

    element_logits2_stack = []
    ap1 = 0
    ap2 = 0
    gts = []
    with torch.no_grad():
        for i, item in enumerate(tqdm(testdataloader)):
        # for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)

            gt_label = 0 if item[1][0] == 'A' else 1
            gts.append(gt_label)

            length = item[2]

            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            if i == 0:
                ap1 = prob1
                ap2 = prob2
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            # element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

    ap1 = ap1.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap1 = ap1.tolist()
    ap2 = ap2.tolist()

    # ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    # AP1 = average_precision_score(gt, np.repeat(ap1, 16))
    # ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    # AP2 = average_precision_score(gt, np.repeat(ap2, 16))

    # print("AUC1: ", ROC1, " AP1: ", AP1)
    # print("AUC2: ", ROC2, " AP2:", AP2)

    # dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    vid_pred = getvidpred(element_logits2_stack)
    AUC = roc_auc_score(gts,vid_pred)
    AP = average_precision_score(gts,vid_pred)
    ACC = accuracy_score(gts,vid_pred)
    # averageMAP = 0
    # for i in range(5):
    #     print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
    #     averageMAP += dmap[i]
    # averageMAP = averageMAP/(i+1)
    # print('average MAP: {:.2f}'.format(averageMAP))
    
    obj=args.obj
    with open("results/VadCLIP/result.txt","a") as f:
        print(f"{obj}", "AUC:{:.3f}".format(AUC), "AP:{:.3f}".format(AP), "ACC:{:.3f}".format(ACC), file=f)
    return AUC, AP ,0#, averageMAP


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = option.parser.parse_args()

    # label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})
    # label_map =  dict({'A': 'normal', 'B1': 'right wheel stuck', 'B2': 'left wheel stuck', 'B3': 'both wheels stuck'})
    label_map =  label_map =  dict({'A': 'normally functioning', 'B': 'fail to funtion'})
    # test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_dataset = DynaDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    # gt = np.load(args.gt_path)
    # gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    # gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
    model_path = os.path.join(args.model_path, f'model_{args.obj}.pth')
    model_param = torch.load(model_path)
    model.load_state_dict(model_param)

    test(model, test_loader, args.visual_length, prompt_text, device)