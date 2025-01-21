# Physics-AD

This is official repository of Physics-AD

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [How to run](#how-to-run)
- [License](#license)

## Overview

This repository is a benchmark for Physics-AD dataset, including unsupervised methods (MemAE, MNAD, MPN, SVM),  weakly-supervised(MGFN, S3R, VadCLIP) and LLM based methods (VideoChatgpt, VideoLLaMA, VideoLLaVA, LAVAD, ZSCLIP, ZSImageBind)


## Data preparation
For all algorithms, in addition to the original data, we also need to prepare the following forms of data:
- frames
- clip features
- i3d features

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
**Note**: our object directory is equivalent to the dataset directory in the official document, and no ground_truth.csv is required.

## Installation

### Install Dependencies

The environmental differences for the algorithm are quite significant. We have provided three environments for the algorithm, corresponding to the following algorithms:

For LAVAD:
```bash
# Install Python dependencies
pip install -r requirements_1.txt
```

For Video-LLaVA:
```bash
# Install Python dependencies
pip install -r requirements_2.txt
```

For the other:
```bash
# Install Python dependencies
pip install -r requirements_0.txt
```

### Download Pretrained-weights

The pre-trained models can be download from [here](https://pan.baidu.com/s/1Oo_SWM0H7AV4Ep7SLKBHPg) (code=6p2s) and should be organized as follows:
```
Physics-AD/
│
├── pretrained-weights/
│   ├── huggingface/
│   ├── llama_vabranch/
│   ├── llama-2-13b-chat/
│   ├── Video-LLaVA-7B-hf/
│   ├── bpe_simple_vocab_16e6.txt.gz
│   ├── imagebind_huge.pth
│   └── tokenizer.model
```

## How to run
Make sure you have installed the right environment and all of the pretrained weights(especially for the LLM methods), and you can run the algorithms from the scripts under ```scripts``` folder.
For most of the methods there is a related option file for setting the parameters under the ```options``` folder and for them the **data path** and the **object to be detected** are two parameters you should modified according to your own setting. The script for LAVAD is a little different, where you need to modified the parameters in the script directly (data path and object are at the very beginning).

All the results will be saved to ```results``` file and the trained models to ```checkpoints``` file.

Given that different methods have varying requirements for inputs, the following methods need some preparation in advance:

### LAVAD
This method uses frame data as input. You need to firstly generate a ```annotations/``` file and a ```test.txt``` under it, which should be like:
```
0000_ball_anomaly_free 0 240 0
0001_ball_anomaly_free 0 240 0
0002_ball_anomaly_free 0 240 0
0003_ball_leak 0 240 0
...
```
where the first column is the name of the video file, and the third is the total frame number of the video. The second and the fourth are just 0.
The ```annotations``` file should at the same level of the ```frames``` file, that is:
```
frame_data/
├─ ball/
│  ├─ training/
│  │  ├─ frames/
│  │  ├─ annotations/
│  │  │  ├─ test.txt
```
After running the script you will find some new files like ```captions```, ```index``` etc. generated. They won't influence the original ```frames``` file.
### MGFN
This method uses ```i3d``` feature. Two lists of train or test video feature paths are required. Take test list for object ```ball``` for example, it should be like this:
```
path_to_your_data/ball/i3d/test/leak0000.npy
path_to_your_data/ball/i3d/test/leak0001.npy
path_to_your_data/ball/i3d/test/leak0002.npy
path_to_your_data/ball/i3d/test/leak0003.npy
path_to_your_data/ball/i3d/test/leak0005.npy
...
```
The final list should be a ```list``` file like ```test_ball.list```.

### VadCLIP
This method uses ```clip``` feature. This method also need two csvs like ```MGFN```, but we have automated this step. You only need to change the feature root path in ```src/VadCLIP/process.py```. If you want to modified to your own setting (e.g. the ratio of normal and abnormal instances) you can also look into ```src/VadCLIP/process.py``` to change them.

### S3R
This method uses ```i3d``` feature. The preparation for this method is relatively complex. You can find the preparation step above in [Data preparation](#data-preparation), i3d feature part.

# License
```
MIT License

Copyright (c) 2025 Chopper233

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
