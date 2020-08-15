# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import torch

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker
from pysot.utils.bbox import center2corner, Center
from pysot.datasets.anchor_target import AnchorTarget

class SiamRPNLatentTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNLatentTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()
        # create anchor target
        self.anchor_target = AnchorTarget()


    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def _get_bbox(self, s_z):
        # imh, imw = image.shape[:2]
        # if len(shape) == 4:
        #     w, h = shape[2]-shape[0], shape[3]-shape[1]
        # else:
        #     w, h = shape
        # context_amount = cfg.TRACK.CONTEXT_AMOUNT
        exemplar_size = cfg.TRACK.EXEMPLAR_SIZE
        # wc_z = w + context_amount * (w+h)
        # hc_z = h + context_amount * (w+h)
        # s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w,h = self.size
        imh, imw = cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def get_training_data(self, bbox):

        cls, delta, delta_weight, overlap = self.anchor_target(
            bbox, cfg.TRAIN.OUTPUT_SIZE, neg=False)
        return {
            'label_cls': torch.tensor(cls),
            'label_loc': torch.tensor(delta),
            'label_loc_weight': torch.tensor(delta_weight),
            'bbox': np.array(bbox)
        }

    def test_get_training_data(self, data, bbox):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        anchors.generate_all_anchors(im_c=cfg.TRAIN.SEARCH_SIZE // 2,
                                     size=cfg.TRAIN.OUTPUT_SIZE)
        anchor_center = anchors.all_anchors[1]
        cx, cy, w, h = anchor_center[0], anchor_center[1], \
                       anchor_center[2], anchor_center[3]
        delta = data['label_loc'].data.cpu().detach().numpy()
        tcx = delta[0] * w + cx
        txy = delta[1] * h + cy
        tw = np.exp(delta[2]) * w
        th = np.exp(delta[3]) * h
        print(bbox)
        return bbox

    def update_weights(self, data, cls_features, loc_features):
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()
        cls_grads, loc_grads = self.model.rpn_head.get_grads(
            cls_features, loc_features, label_cls, label_loc, label_loc_weight)
        self.model.rpn_head.update_weights(cls_grads, loc_grads)

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

        # inner loop optimization
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        bbox = self._get_bbox(s_z)
        data = self.get_training_data(bbox)
        # test data
        # bbox_new = self.test_get_training_data(data, bbox)
        # print('error of data:{}'.format(np.linalg.norm(np.array(bbox)-np.array(bbox_new))))
        label_cls = data['label_cls']
        if torch.max(label_cls) == 1:
            self.model.inner_loop_optimization(x_crop, data)

    # def test_get_training_data(self, data, scale_z):
    #     loc = data['label_loc']
    #     pred_bbox = self._convert_bbox(loc, self.anchors)
    #     print(pred_bbox)
    #     pscore = data['label_cls'].view(-1)
    #     best_idx = np.argmax(pscore)
    #     best_idx = len(pscore)//2
    #     best_s = pscore[best_idx]
    #
    #     bbox = pred_bbox[:, best_idx] / scale_z
    #
    #     cx = bbox[0] + self.center_pos[0]
    #     cy = bbox[1] + self.center_pos[1]
    #     # smooth bbox
    #     lr = 1
    #     width = self.size[0] * (1 - lr) + bbox[2] * lr
    #     height = self.size[1] * (1 - lr) + bbox[3] * lr
    #     bbox = [cx - width / 2,
    #             cy - height / 2,
    #             width,
    #             height]
    #     return bbox

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        # for getting the label which is used to obtain gradident
        # bbox0: the bbox on x_crop
        bbox0 = pred_bbox[:, best_idx]
        imh, imw = cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE
        w = width * scale_z
        h = height * scale_z
        cx, cy = imw // 2, imh // 2
        cx = bbox0[0] + cx
        cy = bbox0[1] + cy
        bbox0 = center2corner(Center(cx, cy, w, h))

        return {
                'bbox': bbox,
                'best_score': best_score,
                'bbox0': bbox0,
                'cls_feas': outputs['cls_feas'] if 'cls_feas' in outputs.keys() else None,
                'loc_feas': outputs['loc_feas'] if 'loc_feas' in outputs.keys() else None
        }
