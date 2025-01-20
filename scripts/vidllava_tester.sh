#!/bin/bash
cd ..
PYTHONPATH=/home/lc/Desktop/wza/gy/dyna/Phys-AD CUDA_VISIBLE_DEVICES=6,7 deepspeed src/VideoLLaVA/test.py --task "describ"