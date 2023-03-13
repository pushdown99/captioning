#!/bin/bash
dataset=~/workspaces/dataset/coco/info

cp ${dataset}/captions.json .
cp ${dataset}/c_train.json .
cp ${dataset}/c_val.json .
cp ${dataset}/c_test.json .
cp ${dataset}/c_text.json .
