from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck
from pysot.core.config import cfg
from leo.latent_rpn_last_pos_neg2 import MultiLatentRPN
import torch
from pysot.utils.anchor import Anchors

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        def get_latent_rpn_head(name, **kwargs):
            RPNS = {
                'MultiLatentRPN': MultiLatentRPN
            }
            return RPNS[name](**kwargs)
        # build rpn head
        if cfg.LATENT:
            self.rpn_head = get_latent_rpn_head(cfg.RPN.TYPE,
                                         **cfg.RPN.KWARGS)
        else:
            self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

        self.auto_params, self.auto_modules = self.get_layers_in_autocoder_model()
        self.auto_opt = self.build_opt_autocoder()
        # for nms label
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
                          cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchors = self.generate_anchor(self.score_size)


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

    def set_meta_train(self, is_meta_train):
        if cfg.RPN.TYPE == 'MultiLatentRPN':
            for idx in range(3):
                # decoder
                rpn = getattr(self.rpn_head, 'rpn' + str(idx+2))
                rpn.cls.is_meta_train = is_meta_train
                rpn.loc.is_meta_train = is_meta_train

    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def get_head_layers(self):
        if cfg.RPN.TYPE == 'MultiLatentRPN':
            head1_params = []
            # head
            h1_params = []
            h1_modules = []
            h2_params = []
            h2_modules = []
            for idx in range(2, 5):
                rpn = getattr(self.rpn_head, 'rpn' + str(idx))
                h1_params += list(map(id, rpn.cls.conv_kernel.parameters()))
                h1_params += list(map(id, rpn.cls.conv_search.parameters()))
                h1_params += list(map(id, rpn.loc.conv_kernel.parameters()))
                h1_params += list(map(id, rpn.loc.conv_search.parameters()))

                h1_modules += list(map(id, rpn.cls.conv_kernel.modules()))
                h1_modules += list(map(id, rpn.cls.conv_search.modules()))
                h1_modules += list(map(id, rpn.loc.conv_kernel.modules()))
                h1_modules += list(map(id, rpn.loc.conv_search.modules()))

                h2_params += list(map(id, rpn.cls.head.parameters()))
                h2_params += list(map(id, rpn.loc.head.parameters()))

                h2_modules += list(map(id, rpn.cls.head.modules()))
                h2_modules += list(map(id, rpn.loc.head.modules()))

            return h1_params, h1_modules, h2_params, h2_modules

    def build_opt_autocoder(self):
        # self.set_fixed_params(self.auto_params, self.auto_modules, False)
        trainable_params = []

        # trainable_params += [{'params': filter(lambda x: x.requires_grad,
        #                                        self.rpn_head.parameters()),
        #                       'lr': cfg.TRAIN.BASE_LR}]
        trainable_params += [{'params': self.auto_params,
                              'lr': cfg.TRAIN.BASE_LR}]
        if cfg.MASK.MASK:
            trainable_params += [{'params': self.mask_head.parameters(),
                                  'lr': cfg.TRAIN.BASE_LR}]

        if cfg.REFINE.REFINE:
            trainable_params += [{'params': self.refine_head.parameters(),
                                  'lr': cfg.TRAIN.BASE_LR}]

        optimizer = torch.optim.SGD(trainable_params,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        return optimizer

    def get_layers_in_autocoder_model(self):
        if cfg.RPN.TYPE == 'MultiLatentRPN':
            auto_params = []
            auto_modules = []
            for name, param in self.rpn_head.named_parameters():
                if 'encoder' in name or 'decoder' in name:
                    auto_params.append(param)
                elif 'cls_weight' in name or 'loc_weight' in name:
                    print(name)
                    auto_params.append(param)


            for name, m in self.rpn_head.named_modules():
                if 'encoder' in name or 'decoder' in name:

                    if isinstance(m, nn.BatchNorm2d):
                     auto_modules.append(m)
            return auto_params, auto_modules

    def set_fixed_params(self, fixed_params, fixed_modules, is_fixed):
        if is_fixed:
            for param in fixed_params:
                param.requires_grad = False
            for m in fixed_modules:
                m.eval()
        else:
            for param in fixed_params:
                param.requires_grad = True
            for m in fixed_modules:
                m.train()

    def inner_loop_optimization(self, x, data):
        label_cls = data['label_cls'].cuda()

        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        fea_cls, fea_loc = self.rpn_head(self.zf, xf)

        self.set_fixed_params(self.auto_params, self.auto_modules, False)
        label_cls = self.get_new_label_cls0(fea_cls, fea_loc, label_cls)

        kl = self.rpn_head.update_weights(fea_cls, fea_loc, label_cls)

        self.set_fixed_params(self.auto_params, self.auto_modules, True)
        # for online tracking
        return fea_cls, fea_loc, label_cls

    def track(self, x):
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        if cfg.LATENT:
            cls_features, loc_features = self.rpn_head(self.zf, xf)
            cls, loc = self.rpn_head.get_cls_loc(cls_features, loc_features)
        else:
            cls, loc = self.rpn_head(self.zf, xf)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None,
                'cls_feas': cls_features,
                'loc_feas': loc_features
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def get_new_label_cls(self, cls_features, loc_features, label_cls):
        with torch.no_grad():
            cls0, loc0 = self.rpn_head.get_cls_loc0(cls_features, loc_features)
            cls0 = self.log_softmax(cls0)
            cls0 = cls0.view(-1, 2)
            c3 = cls0.sum(dim=1).view(-1, 1).repeat(1, 2)
            cls0 = cls0 / c3
            cls0 = cls0[:, 0]
            # cls0 = cls0[:,:,:,:,0]
            s0 = label_cls.shape
            # remove used samples

            label = label_cls.view(-1)
            # pos = label.data.eq(1).nonzero().squeeze().cuda()
            used = label.data.ge(-0.5).nonzero().squeeze().cuda()
            cls0 = cls0.view(-1)
            cls0[used] = 0
            cls0 = cls0 * (-1)
            cls0 = cls0.view(s0[0], -1)
            a, sub = cls0.sort(dim=1)
            n_select = 16
            sub = sub[:, :n_select]
            # convert sub to ind
            m = s0[1] * s0[2] * s0[3]
            n = s0[0]
            c = torch.tensor(range(0, m * n, m)).view(n, 1).cuda()
            d = c.repeat(1, n_select)
            ind = sub + d
            ind = ind.view(-1)

            label[ind] = 0
            label_cls = label.view(s0)
        return label_cls

    def get_new_label_cls0(self, cls_features, loc_features, label_cls):
        with torch.no_grad():
            cls0, loc0 = self.rpn_head.get_cls_loc0(cls_features, loc_features)
            cls0 = self.log_softmax(cls0)
            cls0 = cls0.view(-1, 2)
            c3 = cls0.sum(dim=1).view(-1, 1).repeat(1, 2)
            cls0 = cls0 / c3
            cls0 = cls0[:, 0]
            s0 = label_cls.shape
            # remove used samples
            label = label_cls.view(-1)
            used = label.data.ge(-0.5).nonzero().squeeze().cuda()
            cls0 = cls0.view(-1)
            cls0[used] = 0
            cls0 = cls0 * (-1)
            a, sub = cls0.sort()
            n_select = 16
            ind = sub[:n_select]
            label[ind] = 0
            label_cls = label.view(s0)

        return label_cls

    def get_positive_map(self, label_cls):
        pos_map = label_cls.data.eq(1)
        pos_map = torch.sum(pos_map,1).ge(1)
        neg_map = label_cls.data.ge(0)
        neg_map = torch.sum(neg_map,1).ge(1)
        neg_map = neg_map - pos_map
        # label_sum = torch.sum(label_cls,1)
        pos = pos_map.view(-1).nonzero().squeeze().cuda()
        neg = neg_map.view(-1).nonzero().squeeze().cuda()

        return pos, neg

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()
        bbox = data['bbox']

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        if cfg.LATENT:
            cls_features, loc_features = self.rpn_head(zf, xf)
            if cfg.LATENTS.NEW_LABEL:
                label_cls = self.get_new_label_cls(cls_features, loc_features, label_cls)

            kl = self.rpn_head.update_weights(cls_features, loc_features, label_cls)
            cls, loc = self.rpn_head.get_cls_loc(cls_features, loc_features)

        else:
            cls, loc = self.rpn_head(zf, xf)


        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        features = {}
        features['cls'] = cls_features
        features['loc'] = loc_features

        outputs = {}

        outputs['inner_loop_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss# + 0.001*kl
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        if cfg.LATENTS:
            outputs['out_loop_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                    cfg.TRAIN.LOC_WEIGHT * loc_loss

        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss

        return outputs, features

