# CLNet Training Tutorial

### Add PySOT to your PYTHONPATH
```bash
export PYTHONPATH=/path/to/pysot:$PYTHONPATH
```

## Prepare training dataset
Prepare training dataset, detailed preparations are listed in [training_dataset](training_dataset) directory.
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [YOUTUBEBB](https://research.google.com/youtube-bb/)
* [DET](http://image-net.org/challenges/LSVRC/2017/)
* [COCO](http://cocodataset.org)

## Download pretrained baseline
Download the [original model (ResNet 50)](https://drive.google.com/file/d/1lW_Wj__kOsLGw2H4pwn_eCZK5NywALnm/view?usp=sharing) in [SiamRPN++](https://arxiv.org/abs/1812.11703), 
and put it in `pretrained_models` directory

## Training

To train a model (CLNet), run `tools_leo/train_coder_last_pos_neg2_new_data.py` with the desired configs:

```bash
cd experiments/experiments/clnet_r50_l234
# Single node, multiple GPUs:
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=2333 \
    ../../tools_leo/train_coder_last_pos_neg2_new_data.py --cfg config.yaml

# Testing
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
# Evaluation
python ../../tools_leo/eval.py --tracker_path ./results --dataset VOT2019 --num 4 --tracker_prefix 'ch*' >>res.txt
```

