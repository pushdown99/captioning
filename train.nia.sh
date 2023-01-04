#!/bin/bash
cp lib/config.nia lib/config.py
CUDA_VISIBLE_DEVICES=1 nohup python run.py train -n nia -e 40 -v full > train.nia.out &

