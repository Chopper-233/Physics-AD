#!/bin/bash
cd ..
python -m src.S3R.trainval_anomaly_detector --inference --resume "/home/lc/Desktop/wza/gy/dyna/Phys-AD/checkpoints/S3R/vad-1.0/hinge_s3r_i3d_best.pth" --obj "hinge"