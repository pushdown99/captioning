#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nohup python run.py train -n coco -e 40 -v full > train.coco.out &

