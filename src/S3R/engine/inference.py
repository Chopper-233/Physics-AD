import torch
import numpy as np
import os.path as osp

from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score, roc_auc_score, accuracy_score

def inference(dataloader, model, args, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(device)

        # gt = dataloader.dataset.ground_truths
        dataset = args.obj.lower()

        if args.inference:
            video_list = dataloader.dataset.video_list
            result_dict = dict()
        gts = []
        preds = []
        for i, (video, label, macro) in enumerate(dataloader):
            # print(video.size, macro.size)
            video = video.to(device)
            video = video.permute(0, 2, 1, 3)
            gts.append(label.data.cpu().numpy())
            macro = macro.to(device)

            outputs = model(video, macro)

            # >> parse outputs
            logits = outputs['video_scores']

            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)

            if args.inference:
                video_id = video_list[i]
                result_dict[video_id] = logits.cpu().detach().numpy()

            sig = logits
            K=5
            topk_score, _ = torch.topk(sig, K, dim=0)
            topk_score = torch.mean(topk_score)
            preds.append(topk_score.cpu().detach().numpy())
            # pred = torch.cat((pred, sig))

        # if args.inference:
        #     out_dir = f'output/{dataset}'

        #     import pickle
        #     with open(osp.join(out_dir, f'{dataset}_taskaware_results.pickle'), 'wb') as fout:
        #         pickle.dump(result_dict, fout, protocol=pickle.HIGHEST_PROTOCOL)

        # pred = list(pred.cpu().detach().numpy())
        # pred = np.repeat(np.array(pred), 16)

        gts = list(gts)
        fpr, tpr, threshold = roc_curve(gts, preds)
        rec_auc = auc(fpr, tpr)
        score = rec_auc

        with open("result.txt",'a')as f:
            AUC = roc_auc_score(gts, preds)
            AP = average_precision_score(gts, preds)
            ACC = accuracy_score(gts, [0 if i<0.5 else 1 for i in preds])
            print(args.obj, '{:.3f},{:.3f},{:.3f}'.format(AUC,AP,ACC), file=f)
        # score = 0
        return score

