from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

# from pysot.core.config import cfg
# from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from pysot.core.config import cfg
from leo.model_builder_last_pos_neg2 import ModelBuilder


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', default='VOT2019', type=str,
        help='datasets')
parser.add_argument('--config', default='../experiments/clnet_r50_l234/config-vot.yaml', type=str,
        help='config file')
parser.add_argument('--snapshot', default='../pretrained_models/clnet_r50_l234.pth', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
parser.add_argument('--res_path', default='results', type=str,
        help='result path')
args = parser.parse_args()

import os
# os.environ["CUDA_VISIBLE_DEVICE"] = '2'

torch.set_num_threads(1)

def log_features(feas, tb_writer, tb_index, rank):
    fea_cls = feas['cls']
    fea_loc = feas['loc']
    for i in range(len(fea_cls)):
        fea = fea_cls[i].detach()
        s = fea.shape
        fea = fea.view(s[0]*s[1],-1)
        fea = fea.norm(dim=1)
        fea = fea.cpu().numpy()
        fea = np.float64(fea)

        tb_writer.add_histogram('features_cls{}'.format(i),fea,tb_index)

        fea = fea_loc[i].detach()
        s = fea.shape
        fea = fea.view(s[0]*s[1], -1)
        fea = fea.norm(dim=1)
        fea = fea.cpu().numpy()
        tb_writer.add_histogram('features_loc{}'.format(i), fea, tb_index)

def log_last_weights(model, tb_writer, tb_index, rank):

    with torch.no_grad():
        for idx in range(3):
            rpn = getattr(model.rpn_head, 'rpn' + str(idx + 2))
            w = rpn.cls.last_weights
            w0 = (w - rpn.cls.last_weights0).view(-1).data.cpu().numpy()
            b = rpn.cls.last_bias
            b0 = (b - rpn.cls.last_bias0).view(-1).data.cpu().numpy()
            print('layer{}:cls: w min:{}, w max:{}, b min:{}, b max:{}'.format(
                idx+2, w0.min(), w0.max(), b0.min(), b0.max()
            ))

            # tb_writer.add_scalar('cls/b_norm', np.sum(b0) , tb_index)
            # tb_writer.add_histogram('cls/d_weight{}'.format(idx), w, tb_index)
            # tb_writer.add_histogram('cls/d_bias{}'.format(idx), b, tb_index)

            w = rpn.loc.last_weights
            w0 = (w - rpn.loc.last_weights0).cpu().numpy()
            b = rpn.loc.last_bias
            b0 = (b - rpn.loc.last_bias0).cpu().numpy()
            print('layer{}:loc: w min:{}, w max:{}, b min:{}, b max:{}'.format(
                idx + 2, w0.min(), w0.max(), b0.min(), b0.max()
            ))

            # tb_writer.add_histogram('loc/d_weight{}'.format(idx), w0, tb_index)
            # tb_writer.add_histogram('loc/d_bias{}'.format(idx), b0, tb_index)
from tensorboardX import SummaryWriter

def main():
    # tb_writer = SummaryWriter('logs')
    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()
    for name, p in model.named_parameters():
        if 'last_weights0' in name:
            print(name)
            print(p.data.max())
    # build tracker
    tracker = build_tracker(model)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            # if v_idx < 26:
            #     continue
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            # if v_idx < 32:
            #     continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            img_names = video.img_names
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    # print('best_score:', outputs['best_score'])
                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    # print('idx=',idx)
                    # log_last_weights(model, tb_writer, idx, 0)
                    # log_features(feas, tb_writer, tb_idx, rank)
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        # print('lost_best_score:', outputs['best_score'])
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join(args.res_path, args.dataset, model_name,
                    'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        # OPE tracking
        # coumput time cost for the initization
        init_time = 0
        total_time = 0
        total_frames = 0
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join(args.res_path, args.dataset, model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join(args.res_path, args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join(args.res_path, args.dataset, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))
            init_time+=track_times[0]
            total_time+=toc
            total_frames+=idx
            print('Average init time:{:5.4f}s Average Speed: {:3.1f}fps'.format( init_time/(v_idx+1), total_frames/total_time))

import random
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    seed_torch(233)
    main()
