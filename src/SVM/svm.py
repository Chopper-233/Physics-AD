import numpy as np
from sklearn import svm
import os
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from options.SVM.option import Options


parser = Options().initialize()
args = parser.parse_args()

feat_path = args.feat_path
obj = args.obj

with open("results/SVM/result.txt","a") as f:

    train_features = []
    test_features = []
    gt = []
    for feat in os.listdir(os.path.join(feat_path, obj, 'train')):
        feature = np.load(os.path.join(feat_path,obj,'train',feat))
        train_features.append(feature.reshape(feature.shape[0],-1))
    
    for anmtp in os.listdir(os.path.join(feat_path, obj, 'test')):
        for feat in os.listdir(os.path.join(feat_path, obj, 'test', anmtp)):
            feature = np.load(os.path.join(feat_path,obj,'test',anmtp,feat))
            test_features.append(feature.reshape(feature.shape[0],-1))
            # print(feature.shape, anmtp, feat)
            gt.append(0 if anmtp == "anomaly_free" else 1)
            

    # 加载预先提取的特征（假设每个样本为一行，特征为列）
    train_features = np.concatenate(train_features, axis=0)  # 正常视频的特征
    # test_features = np.concatenate(test_features, axis=0)  # 待检测视频的特征

    # 创建并训练 One-Class SVM 模型
    clf = svm.OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)
    clf.fit(train_features)
    predictions=[]
    for test_feat in test_features:
        # 使用模型进行预测
        # print(test_feat)
        predictions.append(clf.predict(test_feat))

        # 输出检测结果
        # -1 表示异常，1 表示正常
    predictions = ((-1*np.array(predictions)+1)/2)[:,0].tolist()
    # seudo_pred = [1 for _ in range(len(gt))]
    auc = roc_auc_score(gt, predictions)
    ap = average_precision_score(gt, predictions)
    acc = accuracy_score(gt, predictions)

    auc = roc_auc_score(gt, seudo_pred)
    ap = average_precision_score(gt, seudo_pred)
    acc = accuracy_score(gt, seudo_pred)

    print(obj, "AUC: {:.3f}, AP: {:.3f}, ACC: {:.3f}".format(auc,ap,acc), file=f)
