# CLNet-tracking
Implementation code for

CLNet: A Compact Latent Network for Fast Adjusting Siamese Trackers. In Proceedings of the European Conference on Computer Vision (ECCV), 2020.

By Xingping Dong, Jianbing Shen, Ling Shao, Fatih Porikli.

========================================================================

Any comments, please email: xingping.dong@gmail.com, shenjianbingcg@gmail.com

This software was based on the [PySOT](https://github.com/STVIR/pysot) and developed under Ubuntu 16.04 with python 3.7.

If you use this software for academic research, please consider to cite the following papers:


```
@inproceedings{dong2020clnet,
  title={CLNet: A Compact Latent Network for Fast Adjusting Siamese Trackers},
  author={Dong, Xingping and Shen, Jianbing and Shao, Ling and Porikli, Fatih},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
@article{dong2019dynamical,
  title={Dynamical Hyperparameter Optimization via Deep Reinforcement Learning in Tracking},
  author={Dong, Xingping and Shen, Jianbing and Wang, Wenguan and Shao, Ling and Ling, Haibin and Porikli, Fatih},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2019},
  publisher={IEEE}
}
@inproceedings{dong2018hyperparameter,
  title={Hyperparameter optimization for tracking with continuous deep q-learning},
  author={Dong, Xingping and Shen, Jianbing and Wang, Wenguan and Liu, Yu and Shao, Ling and Porikli, Fatih},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={518--527},
  year={2018}
}
@article{dong2019quadruplet,
  title={Quadruplet network with one-shot learning for fast visual object tracking},
  author={Dong, Xingping and Shen, Jianbing and Wu, Dongming and Guo, Kan and Jin, Xiaogang and Porikli, Fatih},
  journal={IEEE Transactions on Image Processing},
  volume={28},
  number={7},
  pages={3516--3527},
  year={2019},
  publisher={IEEE}
}
@inproceedings{dong2018triplet,
  title={Triplet loss in siamese network for object tracking},
  author={Dong, Xingping and Shen, Jianbing},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={459--474},
  year={2018}
}
```
## Installation

Please find installation instructions for PyTorch and PySOT in [`INSTALL.md`](INSTALL.md).

## Pretrained Models and Raw results
We provide the [original model (ResNet 50)](https://drive.google.com/file/d/1lW_Wj__kOsLGw2H4pwn_eCZK5NywALnm/view?usp=sharing) in [SiamRPN++](https://arxiv.org/abs/1812.11703) as the base model for training, 
and the [pretrained model](https://drive.google.com/file/d/1u6O2wqnlWZiuBuwxvzNLqC-knnCJhTRO/view?usp=sharing) of CLNet for testing. 
Please download the above models and put them in the folder `./pretrained_models`.

We also obtain the [raw results](https://drive.google.com/drive/folders/1-MDTMkH6TLjGhLwuH9nqMLX9D5FdWEy4?usp=sharing) for comparison with other methods.

## Testing

### Add PySOT to your PYTHONPATH
```bash
export PYTHONPATH=/path/to/pysot:$PYTHONPATH
```
### Download models
Download the [pretrained model](https://drive.google.com/file/d/1u6O2wqnlWZiuBuwxvzNLqC-knnCJhTRO/view?usp=sharing) of CLNet, 
and put it in the folder `./pretrained_models`.

### Webcam demo
```bash
python tools_leo/demo.py \
    --config experiments/clnet_r50_l234/config-vot.yaml \
    --snapshot pretrained_models/clnet_r50_l234.pth
    # --video demo/bag.avi # (in case you don't have webcam)
```

### Download testing datasets
Download datasets and put them into `testing_dataset` directory. Jsons of OTB, VOT2016/2018/2019, NFS, UAV, LaSOT datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI) or [BaiduYun](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA#list/path=%2F). 
Jsons of DTB, Temple-color-128 datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/15ShGVLdenuERiYAPvipSSY-gT-y3gsNN?usp=sharing). 
More details could be found in `./testing_dataset/README.md`.
If you want to test tracker on new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to setting `testing_dataset`. 

### Test tracker
```bash
cd experiments/clnet_r50_l234
python -u ../../tools_leo/test_last_pos_neg2.py	\
	--snapshot ../../pretrained_models/clnet_r50_l234.pth 	\ # model path
	--dataset VOT2019 	\ # dataset name
	--config config-vot.yaml	  # config file
```
The testing results will in the current directory(results/dataset/model_name/)

### Eval tracker
assume still in experiments/clnet_r50_l234
``` bash
python ../../tools_leo/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset VOT2019        \ # dataset name
	--num 1 		 \ # number thread to eval
	--tracker_prefix 'cl*'   # tracker_name
```

## Training
See [TRAIN.md](TRAIN.md) for detailed instruction.

## Reproduction
We obtain the [raw results](https://drive.google.com/drive/folders/1-MDTMkH6TLjGhLwuH9nqMLX9D5FdWEy4?usp=sharing) for comparison with other methods.

To reproduce our results, we also provide the docker image in [docker hub](https://hub.docker.com/) to run our codes. Please see `experiments/clnet_r50_l234/README.md` for more details.
