#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

datapath=./data/images  # for mini
#datapath=./data/tiered-imagenet/data  # for tiered
#datapath=./data/CUB/CUB_200_2011/images # for CUB

## Change the network name accordingly
config=./configs/mini/resnet12.config # mini
#config=./configs/tiered/resnet18.config # tiered
#config=./configs/cub/resnet18.config # cub
#
python ./src/train.py -c $config --data $datapath --log-file /poodle.log --evaluate
