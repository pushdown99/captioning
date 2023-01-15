#!/bin/bash
cp lib/config.coco lib/config.py
CUDA_VISIBLE_DEVICES=0,1,2 nohup python run.py train --data=coco > train.coco.out &

