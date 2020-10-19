#!/bin/bash

python -u ../../tools_leo/test_last_pos_neg2.py \
        --snapshot ../../pretrained_models/clnet_r50_l234.pth \
	--config config-nfs.yaml \
	--dataset NFS30 
python ../../tools_leo/eval.py --tracker_path ./results --dataset NFS30 --num 4 --tracker_prefix 'c*' >>res-nfs.txt
