from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.xcorr import xcorr_depthwise
from leo.auto_coder_last_pos_neg2 import EncoderCls, EncoderLoc, Decoder, EncoderLinear
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.core.config import cfg

class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError


class LatentDepthwiseXCorrCls(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, n_latent=128, de_hidden = 128,is_meta_training=True):
        super(LatentDepthwiseXCorrCls, self).__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            # nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

        if cfg.LATENTS.ENCODER_ADJUST == '0-layer':
            n_latent = hidden
        if cfg.LATENTS.DECODER_NAME == 'linear':
            self.encoder = EncoderLinear(hidden, n_latent)
        else:
            self.encoder = EncoderCls(hidden, n_latent)
        if cfg.LATENTS.DECODER_WIDE:
            de_hidden = de_hidden * 2
        if cfg.LATENTS.DECODER_BIAS:
            self.decoder = Decoder(n_latent*4, de_hidden, hidden, out_channels, bias=True)
        else:
            self.decoder = Decoder(n_latent * 4, de_hidden, hidden, out_channels)
        self.reconstruct_loss = nn.L1Loss()
        self.is_meta_train = False
        self.last_weights0 = nn.Parameter(torch.zeros(out_channels, hidden, 1, 1))
        self.last_bias0 = nn.Parameter(torch.zeros(out_channels))
        self.last_weights = None #torch.zeros(out_channels, hidden, 1, 1).cuda()#nn.Parameter(torch.zeros(out_channels, hidden, 1, 1))
        self.last_bias = None #torch.zeros(out_channels).cuda()  #nn.Parameter(torch.zeros(out_channels))
        self.layer_weight = torch.zeros(1).cuda()

    def update_weight(self, input, label_cls):
        latents, kl, s1 = self.encoder(input, label_cls)
        weights, bias, layer_weight = self.decoder(latents, s1)
        self.last_weights = self.last_weights0.data + weights
        self.last_bias = self.last_bias0.data + bias
        self.layer_weight = layer_weight
        return kl

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)

        feature = xcorr_depthwise(search, kernel)
        out0 = self.head(feature)

        if self.last_weights is None:
            self.last_weights = self.last_weights0.data
            self.last_bias = self.last_bias0.data

        return out0

class LatentDepthwiseXCorrLoc(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, n_latent=256, de_hidden = 128,is_meta_training=True):
        super(LatentDepthwiseXCorrLoc, self).__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            # nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

        if cfg.LATENTS.ENCODER_ADJUST == '0-layer':
            n_latent = hidden

        if cfg.LATENTS.DECODER_NAME == 'linear':
            self.encoder = EncoderLinear(hidden, n_latent // 2)
        else:
            self.encoder = EncoderLoc(hidden, n_latent)

        if cfg.LATENTS.DECODER_WIDE:
            de_hidden = de_hidden * 2
        if cfg.LATENTS.DECODER_BIAS:
            self.decoder = Decoder(n_latent * 2, de_hidden, hidden, out_channels, bias=True)
        else:
            self.decoder = Decoder(n_latent * 2, de_hidden, hidden, out_channels)
        self.reconstruct_loss = nn.L1Loss()
        self.is_meta_train = False
        self.last_weights0 = nn.Parameter(torch.zeros(out_channels, hidden, 1, 1))
        self.last_bias0 = nn.Parameter(torch.zeros(out_channels))
        self.last_weights = None #torch.zeros(out_channels, hidden, 1, 1).cuda()#nn.Parameter(torch.zeros(out_channels, hidden, 1, 1))
        self.last_bias = None #torch.zeros(out_channels).cuda()  #nn.Parameter(torch.zeros(out_channels))
        self.layer_weight = torch.zeros(1).cuda()

    def update_weight(self, input, label_cls):
        latents, kl, s1 = self.encoder(input, label_cls)
        weights, bias, layer_weight = self.decoder(latents, s1)

        self.last_weights = self.last_weights0.data + weights
        self.last_bias = self.last_bias0.data + bias
        self.layer_weight = layer_weight

        return kl

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)

        feature = xcorr_depthwise(search, kernel)
        out0 = self.head(feature)

        if self.last_weights is None:
            self.last_weights = self.last_weights0.data
            self.last_bias = self.last_bias0.data

        return out0

class LatentDepthwiseRPN(RPN):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):
        super(LatentDepthwiseRPN, self).__init__()
        self.cls = LatentDepthwiseXCorrCls(in_channels, out_channels, 2 * anchor_num)
        self.loc = LatentDepthwiseXCorrLoc(in_channels, out_channels, 4 * anchor_num)

    def forward(self, z_f, x_f):
        cls_fea= self.cls(z_f, x_f)
        loc_fea = self.loc(z_f, x_f)
        return cls_fea, loc_fea

class MultiLatentRPN(RPN):
    def __init__(self, anchor_num, in_channels, weighted=False):
        super(MultiLatentRPN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('rpn'+str(i+2),
                    LatentDepthwiseRPN(anchor_num, in_channels[i], in_channels[i]))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def avg(self, lst):
        return sum(lst) / len(lst)

    def weighted_avg(self, lst, weight):
        s = 0
        for i in range(len(weight)):
            s += lst[i] * weight[i]
        return s

    def get_grads(self, cls_feas, loc_feas, label_cls, label_loc, label_loc_weight):
        cls = []
        loc = []
        cls_lws, loc_lws = [], []
        for idx, (cls_fea, loc_fea) in enumerate(zip(cls_feas, loc_feas), start=2):
            rpn = getattr(self, 'rpn' + str(idx))
            cls_fea = cls_fea.data.detach()
            cls_fea.requires_grad = True
            c = F.conv2d(cls_fea, weight=rpn.cls.last_weights.detach(), bias=rpn.cls.last_bias.detach())
            loc_fea = loc_fea.data.detach()
            loc_fea.requires_grad = True
            l = F.conv2d(loc_fea, weight=rpn.loc.last_weights.detach(), bias=rpn.loc.last_bias.detach())
            cls.append(c)
            loc.append(l)
            cls_feas[idx - 2] = cls_fea
            loc_feas[idx - 2] = loc_fea

            clw = rpn.cls.layer_weight
            llw = rpn.loc.layer_weight
            cls_lws.append(clw)
            loc_lws.append(llw)

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight + torch.cat(cls_lws).detach(), 0)
            loc_weight = F.softmax(self.loc_weight + torch.cat(loc_lws).detach(), 0)

        if self.weighted:
            cls, loc = self.weighted_avg(cls, cls_weight), self.weighted_avg(loc, loc_weight)
        else:
            cls, loc = self.avg(cls), self.avg(loc)

        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)
        loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        loss.backward()

        cls_grads = []
        loc_grads = []
        for idx, (cls_fea, loc_fea) in enumerate(zip(cls_feas, loc_feas)):
            cls_grads.append(cls_fea.grad.data.detach()*10000)
            loc_grads.append(loc_fea.grad.data.detach()*10000)

        return cls_grads, loc_grads

    def update_weights(self, cls_feas, loc_feas, label_cls):
        kl = 0
        for idx, (cls_fea, loc_fea) in enumerate(zip(cls_feas, loc_feas), start=2):
            rpn = getattr(self, 'rpn' + str(idx))
            kl1 = rpn.cls.update_weight(cls_fea, label_cls)
            kl2 = rpn.loc.update_weight(loc_fea, label_cls)
            kl = kl + kl1 + kl2
        return kl

    def get_cls_loc(self, cls_feas, loc_feas):
        cls = []
        loc = []
        cls_lws, loc_lws = [], []
        for idx, (cls_fea, loc_fea) in enumerate(zip(cls_feas, loc_feas), start=2):
            rpn = getattr(self, 'rpn' + str(idx))
            c = F.conv2d(cls_fea, weight=rpn.cls.last_weights, bias=rpn.cls.last_bias)
            l = F.conv2d(loc_fea, weight=rpn.loc.last_weights, bias=rpn.loc.last_bias)
            cls.append(c)
            loc.append(l)
            clw = rpn.cls.layer_weight
            llw = rpn.loc.layer_weight
            cls_lws.append(clw)
            loc_lws.append(llw)

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight + torch.cat(cls_lws), 0)
            loc_weight = F.softmax(self.loc_weight + torch.cat(loc_lws), 0)
            return self.weighted_avg(cls, cls_weight), self.weighted_avg(loc, loc_weight)
        else:
            return self.avg(cls), self.avg(loc)

    def get_cls_loc0(self, cls_feas, loc_feas):
        cls = []
        loc = []
        for idx, (cls_fea, loc_fea) in enumerate(zip(cls_feas, loc_feas), start=2):
            rpn = getattr(self, 'rpn' + str(idx))
            c = F.conv2d(cls_fea, weight=rpn.cls.last_weights0, bias=rpn.cls.last_bias0)
            l = F.conv2d(loc_fea, weight=rpn.loc.last_weights0, bias=rpn.loc.last_bias0)
            cls.append(c)
            loc.append(l)

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)
            return self.weighted_avg(cls, cls_weight), self.weighted_avg(loc, loc_weight)
        else:
            return self.avg(cls), self.avg(loc)

    def forward(self, z_fs, x_fs):

        cls_feas = []
        loc_feas = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            rpn = getattr(self, 'rpn'+str(idx))
            cls_fea, loc_fea = rpn(z_f, x_f)

            cls_feas.append(cls_fea)
            loc_feas.append(loc_fea)

        return cls_feas, loc_feas