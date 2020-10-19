#!bin/bash

source /software/conda/etc/profile.d/conda.sh

conda activate pysot

export PYTHONPATH=code_path_in_container:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=2333 \
    ../../tools_leo/train_coder_last_pos_neg2_new_data.py --cfg config.yaml


START=1
END=20
N=4
device_start=0
for i in $(seq $START  $END)
do
{
    echo $i
    device=$(($i % $N))
    device=$(($device_start + $device))
    echo $device
    export CUDA_VISIBLE_DEVICES=$device

    python -u ../../tools_leo/test_last_pos_neg2.py \
        --snapshot "snapshot/checkpoint_e"$i".pth" \
	--config config.yaml \
	--res_path results \
	--dataset VOT2019 2>&1 | tee "logs/test_dataset"$i".log"
}&
done
wait

python ../../tools_leo/eval.py --tracker_path ./results --dataset VOT2019 --num 4 --tracker_prefix 'ch*' >>res.txt
