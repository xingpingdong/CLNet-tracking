#!/bin/bash

python -u ../../tools_leo/test_last_pos_neg2.py \
        --snapshot ../../pretrained_models/clnet_r50_l234.pth \
	--config config-uav.yaml \
	--dataset UAV123 
python ../../tools_leo/eval.py --tracker_path ./results --dataset UAV123 --num 4 --tracker_prefix 'c*' >>res-uav.txt
