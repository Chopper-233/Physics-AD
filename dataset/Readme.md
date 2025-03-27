
### Frames
Split the video into frames and organize them as following:
```
frame_data/
├─ ball/
│  ├─ training/
│  │  ├─ frames/
│  │  │  ├─ 0000_ball_train/
│  │  │  │  ├─ frame_0001.png
│  │  │  │  ├─ frame_0002.png
│  │  │  │  ├─ ...
│  │  │  ├─ 0001_ball_train/
│  │  │  ├─ ...
│  ├─ testing/
│  │  ├─ frames/
│  │  │  ├─ 0000_ball_leak/
│  │  │  │  ├─ frame_0001.png
│  │  │  │  ├─ frame_0002.png
│  │  │  │  ├─ ...
│  │  │  ├─ 0001_ball_leak/
│  │  │  ├─ ...
├─ button/
├─ ...
```

### Clip features
Extract clip features and organize them as following:
```
clipfeature/
├─ TrainClipFeatures
│  ├─ ball/
│  │  ├─ ball_train_0000.npy
│  │  ├─ ball_train_0001.npy
│  │  ├─ ...
│  ├─ button/
│  ├─ ...
├─ TestClipFeatures
│  ├─ ball/
│  │  ├─ ball_test_anomaly_free_0000.npy
│  │  ├─ ball_test_leak_0014.npy
│  │  ├─ ...
│  ├─ button/
│  ├─ ...
```
### I3d features
I3d features are extracted using pytorch-i3d and organized follwing the S3R method. You can find the original official preprocessing method [here](https://github.com/louisYen/S3R). Finally it should be like this:
```
S3R_data/
├─ dictionary/
│  ├─ ball/
│  │  ├─ ball_dictionaries.taskaware.omp.100iters.90pct.npy
│  │  ├─ ball_regular_features-2048dim.training.pickle
│  ├─ button/
│  ├─ ...
├─ data/
│  ├─ ball/
│  │  ├─ ball.training.csv
│  │  ├─ ball.testing.csv
│  │  ├─ i3d/
│  │  │  ├─ test/
│  │  │  │  ├─anomaly_free0000.npy
│  │  │  │  ├─anomaly_free0001.npy
│  │  │  │  ├─ ...
│  │  │  ├─ train/
│  │  │  │  ├─ train0000.npy
│  │  │  │  ├─ train0001.npy
│  │  │  │  ├─ ...
│  ├─ button/
│  ├─ ...
```
**Note**: 
1. Our object directory is equivalent to the dataset directory in the official document, and no ground_truth.csv is required.
2. If you are running weakly supervised method, please remember to put some anomaly samples into the training set (about 10% of the total anomaly samples in our experiments.)

### Download Pretrained-weights

The pre-trained models can be download from [here](https://pan.baidu.com/s/1Oo_SWM0H7AV4Ep7SLKBHPg) (code=6p2s) (updating now) and should be organized as follows:
```
Phys-AD/
│
├── pretrained-weights/
│   ├── bert-base-uncased/
│   ├── huggingface/
│   ├── llama_vabranch/
│   ├── llama-2-7b-chat/
│   ├── llama-2-13b-chat/
│   ├── Video-LLaVA-7B-hf/
│   ├── bpe_simple_vocab_16e6.txt.gz
│   ├── blip2_pretrained_flant5xxl
│   ├── imagebind_huge.pth
│   └── tokenizer.model
```
