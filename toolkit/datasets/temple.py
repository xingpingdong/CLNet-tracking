import os
import cv2
import json
import numpy as np

from glob import glob
from tqdm import tqdm
from PIL import Image

from .dataset import Dataset
from .video import Video


class TempleVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
    """

    def __init__(self, name, root, video_dir, init_rect, img_names, gt_rect, load_img=False):
        super(TempleVideo, self).__init__(name, root, video_dir,
                                          init_rect, img_names, gt_rect, None, load_img)
        self.tags = {'all': [1] * len(gt_rect)}

        if not load_img:
            img_name = os.path.join(root, self.img_names[0])
            img = np.array(Image.open(img_name), np.uint8)
            self.width = img.shape[1]
            self.height = img.shape[0]

    # def load_tracker(self, path, tracker_names=None, store=True):
    #     """
    #     Args:
    #         path(str): path to result
    #         tracker_name(list): name of tracker
    #     """
    #     if not tracker_names:
    #         tracker_names = [x.split('/')[-1] for x in glob(path)
    #                          if os.path.isdir(x)]
    #     if isinstance(tracker_names, str):
    #         tracker_names = [tracker_names]
    #     for name in tracker_names:
    #         traj_files = glob(os.path.join(path, name, 'baseline', self.name, '*0*.txt'))
    #         if len(traj_files) == 15:
    #             traj_files = traj_files
    #         else:
    #             traj_files = traj_files[0:1]
    #         pred_traj = []
    #         for traj_file in traj_files:
    #             with open(traj_file, 'r') as f:
    #                 traj = [list(map(float, x.strip().split(',')))
    #                         for x in f.readlines()]
    #                 pred_traj.append(traj)
    #         if store:
    #             self.pred_trajs[name] = pred_traj
    #         else:
    #             return pred_traj


class TempleDataset(Dataset):
    """
    Args:
        dataset_root: dataset root
        load_img: wether to load all imgs
    """

    def __init__(self, name, dataset_root, load_img=False):
        super(TempleDataset, self).__init__(name, dataset_root)
        with open(os.path.join(dataset_root, name + '.json'), 'r') as f:
            meta_data = json.load(f)

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading Temple-color-128', ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = TempleVideo(video,
                                             dataset_root,
                                             meta_data[video]['video_dir'],
                                             meta_data[video]['init_rect'],
                                             meta_data[video]['img_names'],
                                             meta_data[video]['gt_rect'],
                                             load_img=load_img)

        self.tags = ['all']


