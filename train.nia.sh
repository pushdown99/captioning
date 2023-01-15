#!/bin/bash
cp lib/config.nia lib/config.py
CUDA_VISIBLE_DEVICES=0,1,2 nohup python run.py train --data=nia > train.nia.out &

