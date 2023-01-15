#!/bin/bash
if [ $# -ne 1 ]; then
  echo "Usage: $0 {pretrained model path}"
  exit -1
else
  cp lib/config.nia lib/config.py
  CUDA_VISIBLE_DEVICES=0,1,2 nohup python run.py eval --trained=$1 > train.nia.out &
fi

