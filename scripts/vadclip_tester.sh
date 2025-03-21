#!/bin/bash
cd ..
python -m src.VadCLIP.process --obj "hinge"
python -m src.VadCLIP.test --obj "hinge"