<div align="center">

# Towards Visual Discrimination and Reasoning of Real-World Physical Dynamics: Physics-Grounded Anomaly Detection
# CVPR 2025

<div align="center" margin-bottom="6em">
    <span class="author-block">
        <a href="" target="_blank">Wenqiao Li✶</a><sup>1</sup>,</span>
    <span class="author-block">
        <a href="" target="_blank">Yao Gu✶</a><sup>1</sup>,</span>
    <span class="author-block">
        <a href="" target="_blank">Xintao Chen✶</a><sup>1</sup>,</span>
    <span class="author-block">
        <a href="https://scholar.google.com/citations?hl=en&user=3Ifn2DoAAAAJ&view_op=list_works&sortby=pubdate" target="_blank">Xiaohao Xu</a><sup>2</sup>,</span>
    <span class="author-block">
        <a href="" target="_blank">Ming Hu</a><sup>3</sup>,</span>
    <span class="author-block">
        <a href="https://robotics.umich.edu/people/faculty/xiaonan-sean-huang/" target="_blank">Xiaonan Huang</a><sup>2</sup></span>
     <span class="author-block">
        <a href="" target="_blank">Yingna Wu</a><sup>1</sup></span>
    <br>
    <p style="font-size: 0.9em; padding: 0.5em 0;">✶ indicates equal contribution</p>
    <span class="author-block">
        <sup>1</sup>ShanghaiTech University &nbsp&nbsp 
        <sup>2</sup>University of Michigan, Annor bor &nbsp&nbsp 
        <sup>3</sup>Monash University
    </span>

[Website]() | [Arxiv]() | [Data]()
</div>
</div>


## Table of Contents

- [Overview](#overview)
- [Preparation](#preparation)
- [Installation](#installation)
- [How to run](#how-to-run)
- [Links to methods](#links-to-methods)
- [PAEval](#paeval)
- [Citation](#citation)
- [License](#license)

## Overview

This repository is a benchmark for Phys-AD dataset, including unsupervised methods (MemAE, MNAD, MPN, SVM),  weakly-supervised(MGFN, S3R, VadCLIP) and LLM based methods (VideoChatgpt, VideoLLaMA, VideoLLaVA, LAVAD, ZSCLIP, ZSImageBind)


## Preparation

For all algorithms, in addition to the original data, we also need to prepare the following forms of data:
- frames
- clip features
- i3d features
 
For details and pre-trained weights downloading, please refer to [here](./dataset/Readme.md).

For some methods, some extra pre-process should be applied, please refer to [here](./src/Readme.md).

## Installation

### Install Dependencies

The environmental differences for the algorithm are quite significant. We have provided three environments for the algorithm, corresponding to the following algorithms:


#### For most methods included:
```bash
# Install Python dependencies
pip install -r requirements.txt
```
#### While there are 2 exceptions:  
For LAVAD:
```bash
# Install Python dependencies
pip install -r requirements_lavad.txt
```

For Video-LLaVA:
```bash
# Install Python dependencies
pip install -r requirements_llava.txt
```


## How to run
Make sure you have installed the right environment and all of the pretrained weights(especially for the LLM methods), and you can run the algorithms from the scripts under ```scripts``` folder.  
For most of the methods there are related option files to set the parameters under the ```options``` folder. Among the params the **data path** and the **object to be detected** are two parameters you should modified according to your own setting. 

**Note**: The script for **LAVAD** is a little different, where you need to modified the parameters in the script directly (data path and object are at the very beginning).

```bash
cd scripts
sh script_of_method_you_want_to_run.sh
```
### Unsurpervised methods
**For example**, if you want to train and test **MemAE** method:  
1. Switch to the ```scripts``` folder.  
```bash
cd scripts
```
2. Find the scripts of the method you want to run. You may want to modify the flag ```--obj``` in the script to specify the object you want to train or test.  
3. For training:
```bash
sh memae_trainer.sh
```
4. For testing:
```bash
sh memae_tester.sh
```
### Weakly-surpervised methods
**For example**, if you want to train and test **VadCLIP** method:  

0. Randomly put some abnormal samples into the training set (10% of the total abnormal samples in our experiments). 
1. Switch to the ```scripts``` folder.  
```bash
cd scripts
```
2. Find the scripts of the method you want to run. You may want to modify the flag ```--obj``` in the script to specify the object you want to train or test.  
3. For training:
```bash
sh vadclip_trainer.sh
```
4. For testing:
```bash
sh vadclip_tester.sh
```
### Video-understanding methods
**For example**, if you want to test **VideoLLaVA** method:  
1. Switch to the ```scripts``` folder.  
```bash
cd scripts
```
2. Find the scripts of the method you want to run. You may want to modify the flag ```--obj``` in the script to specify the object you want to test. **Check the cuda device setting**.
3. For testing:
```bash
sh videollava_tester.sh
```
**Note**: 
1. In this project we use '_' to connect the name of an object, e.g.: 'rolling_bearing' for 'rolling bearing'.
2. Video understanding methods have only tester and no need to train. 
3. All the results will be saved to ```results``` file and the trained models to ```checkpoints``` file.

## PAEval
Please refer to [here](PAEval/Readme.md) to check more instructions on PAEval experiments.

## Links to methods
[MemAE](https://github.com/donggong1/memae-anomaly-detection)\
[MNAD](https://github.com/cvlab-yonsei/MNAD)\
[MPN](https://github.com/ktr-hubrt/MPN)\
[MGFN](https://github.com/carolchenyx/MGFN.)\
[S3R](https://github.com/louisYen/S3R)\
[VadCLIP](https://github.com/nwpu-zxr/VadCLIP)\
[ZSCLIP](https://github.com/openai/CLIP)\
[ZSImageBind](https://github.com/facebookresearch/ImageBind)\
[VideoChatgpt](https://github.com/OpenGVLab/Ask-Anything)\
[VideoLLaMA](https://github.com/DAMO-NLP-SG/VideoLLaMA2)\
[VideoLLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA)\
[LAVAD](https://github.com/lucazanella/lavad)


# Citation
Please cite the following paper if this work helps your project:
```bibtex
@article{li2025towards,
  title={Towards Visual Discrimination and Reasoning of Real-World Physical Dynamics: Physics-Grounded Anomaly Detection},
  author={Li, Wenqiao and Gu, Yao and Chen, Xintao and Xu, Xiaohao and Hu, Ming and Huang, Xiaonan and Wu, Yingna},
  journal={arXiv preprint arXiv:2503.03562},
  year={2025}
}
```
# License
```
MIT License

Copyright (c) 2025 Phys-AD

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
