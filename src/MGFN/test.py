from torch.utils.data import DataLoader
import options.MGFN.option as option
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, accuracy_score
from tqdm import tqdm
args=option.parse_args()
from src.MGFN.config import *
from models.MGFN.mgfn import mgfn as Model
from dataset.mgfn_dataset import Dataset


def test(dataloader, model, args, device):
    plt.clf()
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        featurelen =[]
        gts = []
        preds = []
        # for i, inputs in enumerate(dataloader):
        with open('results/MGFN/result.txt','a') as f:
            for i, inputs in enumerate(tqdm(dataloader)):

                input = inputs[0].to(device)
                label = inputs[2]
                gts.append(label.data.cpu().numpy())
                input = input.permute(0, 2, 1, 3)
                _, _, _, _, logits = model(input)
                logits = torch.squeeze(logits, 1)
                logits = torch.mean(logits, 0)
                sig = logits
                featurelen.append(len(sig))

                K=3
                topk_score, _ = torch.topk(sig, K, dim=0)
                topk_score = torch.mean(topk_score)
                preds.append(topk_score.cpu().detach().numpy())

                # pred = torch.cat((pred, sig))

            # gt = np.load(args.gt)
            # pred = list(pred.cpu().detach().numpy())
            # pred = np.repeat(np.array(pred), 16)
            fpr, tpr, threshold = roc_curve(list(gts), preds)
            rec_auc = auc(fpr, tpr)
            precision, recall, th = precision_recall_curve(list(gts), preds)
            pr_auc = auc(recall, precision)

            gts = list(gts)
            AUC = roc_auc_score(gts, preds)
            AP = average_precision_score(gts, preds)
            preds_acc = [0 if i < 0.5 else 1 for i in preds]
            ACC = accuracy_score(gts, preds_acc)
            print(args.obj, "{:.3f},{:.3f},{:.3f}".format(AUC,AP,ACC), file=f)
            # print('pr_auc : ' + str(pr_auc), file=f)
            # print('rec_auc : ' + str(rec_auc), file=f)
    return rec_auc, pr_auc

if __name__ == '__main__':
    args = option.parse_args()
    config = Config(args)
    device = torch.device("cuda")
    model = Model()
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)
    model = model.to(device)
    model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(f'/home/lc/Desktop/wza/gy/dyna/bench/MGFN.-main/ckpt/zipper_i3d_[0.001]*15000_4_0.1_mgfn/mgfn1-i3d.pkl').items()})
    auc = test(test_loader, model, args, device)
