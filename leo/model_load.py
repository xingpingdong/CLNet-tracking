# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import torch
import torch.nn as nn


logger = logging.getLogger('global')


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # filter 'num_batches_tracked'
    missing_keys = [x for x in missing_keys
                    if not x.endswith('num_batches_tracked')]
    if len(missing_keys) > 0:
        logger.info('[Warning] missing keys: {}'.format(missing_keys))
        logger.info('missing keys:{}'.format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        logger.info('[Warning] unused_pretrained_keys: {}'.format(
            unused_pretrained_keys))
        logger.info('unused checkpoint keys:{}'.format(
            len(unused_pretrained_keys)))
    logger.info('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, \
        'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters
    share common prefix 'module.' '''
    logger.info('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_pretrain(model, pretrained_path):
    logger.info('load pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path,
                                 map_location=lambda storage, loc: storage.cuda(device))
    model_dict = model.state_dict()



    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model

def load_pretrain_rpn_last(model, pretrained_path):
    logger.info('load pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path,
                                 map_location=lambda storage, loc: storage.cuda(device))
    model_dict = model.state_dict()

    # for k, v in model_dict.items():
    #     print(k)

    # for k, v in pretrained_dict.items():
    #     print(k)
    # 1. filter out unnecessary keys
    # pretrained_dict0 = {k: v for k, v in pretrained_dict.items() if k in model_dict }
    # load last head's weights
    pretrained_dict0 = {}
    for k, v in pretrained_dict.items():
        if 'head.3.weight' in k:
            k = k.replace('head.3.weight','last_weights0')
            # print(k), print(v.shape)
        if 'head.3.bias' in k:
            k = k.replace('head.3.bias','last_bias0')

        pretrained_dict0[k] = v
        if 'head.0.weight' in k:
            k = k.replace('head.0.weight', 'last_head0_weight')
            pretrained_dict0[k] = v
        elif 'head.1.weight' in k:
            k = k.replace('head.1.weight', 'encoder_bn.0.weight')
            pretrained_dict0[k] = v
        elif 'head.1.bias' in k:
            k = k.replace('head.1.bias', 'encoder_bn.0.bias')
            pretrained_dict0[k] = v
        elif 'head.1.running_mean' in k:
            k = k.replace('head.1.running_mean', 'encoder_bn.0.running_mean')
            pretrained_dict0[k] = v
        elif 'head.1.running_var' in k:
            k = k.replace('head.1.running_var', 'encoder_bn.0.running_var')
            pretrained_dict0[k] = v
    # 1. filter out unnecessary keys
    pretrained_dict0 = {k: v for k, v in pretrained_dict0.items() if k in model_dict}
    # for k, v in pretrained_dict0.items():
    #     print(k)
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict0)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model

def get_coder_layers(model, type):
    if type == 'MultiLatentRPN':
        encoder_params = []
        decoder_params = []
        encoder_modules = []
        decoder_modules = []
        for name, param in model.named_parameters():
            if 'encoder' in name:
                encoder_params.append(param)
            elif 'decoder' in name:
                decoder_params.append(param)
        for name, m in model.named_modules():
            if 'encoder' in name:
                encoder_modules.append(m)
            elif 'decoder' in name:
                decoder_modules.append(m)
        # for idx in range(2,5):
        #     rpn = getattr(model.rpn_head, 'rpn'+str(idx))
        #     encoder_params += list(map(id, rpn.cls.encoder.parameters()))
        #     encoder_params += list(map(id, rpn.loc.encoder.parameters()))
        #     decoder_params += list(map(id, rpn.cls.decoder.parameters()))
        #     decoder_params += list(map(id, rpn.loc.decoder.parameters()))
        #
        #     encoder_modules += list(map(id, rpn.cls.encoder.modules()))
        #     encoder_modules += list(map(id, rpn.loc.encoder.modules()))
        #     decoder_modules += list(map(id, rpn.cls.decoder.modules()))
        #     decoder_modules += list(map(id, rpn.loc.decoder.modules()))
        return encoder_params, decoder_params, encoder_modules, decoder_modules

def get_coder_layers0(model, type):
    if type == 'MultiLatentRPN':
        encoder_params = []
        decoder_params = []
        encoder_modules = []
        decoder_modules = []
        # for name, param in model.named_parameters():
        #     if 'encoder' in name:
        #         encoder_params.append(param)
        #     elif 'decoder' in name:
        #         decoder_params.append(param)
        # for name, m in model.named_modules():
        #     if 'encoder' in name:
        #         encoder_modules.append(m)
        #     elif 'decoder' in name:
        #         decoder_modules.append(m)
        for idx in range(2,5):
            rpn = getattr(model.rpn_head, 'rpn'+str(idx))
            encoder_params += list(map(id, rpn.cls.encoder.parameters()))
            encoder_params += list(map(id, rpn.loc.encoder.parameters()))
            decoder_params += list(map(id, rpn.cls.decoder.parameters()))
            decoder_params += list(map(id, rpn.loc.decoder.parameters()))

            encoder_modules += list(map(id, rpn.cls.encoder.modules()))
            encoder_modules += list(map(id, rpn.loc.encoder.modules()))
            decoder_modules += list(map(id, rpn.cls.decoder.modules()))
            decoder_modules += list(map(id, rpn.loc.decoder.modules()))
        return encoder_params, decoder_params, encoder_modules, decoder_modules

def get_head_layers(model, type):
    if type == 'MultiLatentRPN':
        head1_params = []
        # head
        h1_params = []
        h1_modules = []
        h2_params = []
        h2_modules = []
        for idx in range(2, 5):
            rpn = getattr(model.rpn_head, 'rpn' + str(idx))
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

def get_inner_loop_layers(model, type):
    if type == 'MultiLatentRPN':
        inner_params = []

def freeze_pretrain_head(model,type):
    base_params = []
    base_modules = []
    if type == 'MultiLatentRPN':
        encoder_params, decoder_params, encoder_modules, decoder_modules = \
            get_coder_layers0(model,type)
        backbone_params = list(map(id, model.backbone.parameters()))
        base_params = filter(
            lambda p: id(
                p) not in encoder_params + decoder_params \
                      + backbone_params,
            model.parameters())
        # rpn2_cls_encoder_params = list(map(id, model.rpn_head.rpn2.cls.encoder.parameters()))
        # rpn2_cls_decoder_params = list(map(id, model.rpn_head.rpn2.cls.decoder.parameters()))
        # rpn2_loc_encoder_params = list(map(id, model.rpn_head.rpn2.loc.encoder.parameters()))
        # rpn2_loc_decoder_params = list(map(id, model.rpn_head.rpn2.loc.decoder.parameters()))
        # rpn3_cls_encoder_params = list(map(id, model.rpn_head.rpn3.cls.encoder.parameters()))
        # rpn3_cls_decoder_params = list(map(id, model.rpn_head.rpn3.cls.decoder.parameters()))
        # rpn3_loc_encoder_params = list(map(id, model.rpn_head.rpn3.loc.encoder.parameters()))
        # rpn3_loc_decoder_params = list(map(id, model.rpn_head.rpn3.loc.decoder.parameters()))
        # rpn4_cls_encoder_params = list(map(id, model.rpn_head.rpn4.cls.encoder.parameters()))
        # rpn4_cls_decoder_params = list(map(id, model.rpn_head.rpn4.cls.decoder.parameters()))
        # rpn4_loc_encoder_params = list(map(id, model.rpn_head.rpn4.loc.encoder.parameters()))
        # rpn4_loc_decoder_params = list(map(id, model.rpn_head.rpn4.loc.decoder.parameters()))
        # backbone_params = list(map(id, model.backbone.parameters()))
        # base_params = filter(
        #     lambda p: id(
        #         p) not in rpn2_cls_decoder_params + rpn2_cls_encoder_params + rpn2_loc_decoder_params + rpn2_loc_encoder_params \
        #               + rpn3_cls_decoder_params + rpn3_cls_encoder_params + rpn3_loc_decoder_params + rpn3_loc_encoder_params \
        #               + rpn4_cls_decoder_params + rpn4_cls_encoder_params + rpn4_loc_decoder_params + rpn4_loc_encoder_params \
        #               + backbone_params,
        #     model.parameters())
        for p in base_params:
            p.requires_grad = False
        # encoders = filter(
        #     lambda p: id(p) in encoder_params, model.parameters())
        # for p in encoders:
        #     p.requires_grad = False

        # for debug

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print("requires_grad: True ", name)
        #     else:
        #         print("requires_grad: False ", name)

        backbone_modules = list(map(id, model.backbone.modules()))
        base_modules = filter(
            lambda p: id(
                p) not in encoder_modules + decoder_modules \
                      + backbone_modules,
            model.modules())

        # rpn2_cls_encoder_modules = list(map(id, model.rpn_head.rpn2.cls.encoder.modules()))
        # rpn2_cls_decoder_modules = list(map(id, model.rpn_head.rpn2.cls.decoder.modules()))
        # rpn2_loc_encoder_modules = list(map(id, model.rpn_head.rpn2.loc.encoder.modules()))
        # rpn2_loc_decoder_modules = list(map(id, model.rpn_head.rpn2.loc.decoder.modules()))
        # rpn3_cls_encoder_modules = list(map(id, model.rpn_head.rpn3.cls.encoder.modules()))
        # rpn3_cls_decoder_modules = list(map(id, model.rpn_head.rpn3.cls.decoder.modules()))
        # rpn3_loc_encoder_modules = list(map(id, model.rpn_head.rpn3.loc.encoder.modules()))
        # rpn3_loc_decoder_modules = list(map(id, model.rpn_head.rpn3.loc.decoder.modules()))
        # rpn4_cls_encoder_modules = list(map(id, model.rpn_head.rpn4.cls.encoder.modules()))
        # rpn4_cls_decoder_modules = list(map(id, model.rpn_head.rpn4.cls.decoder.modules()))
        # rpn4_loc_encoder_modules = list(map(id, model.rpn_head.rpn4.loc.encoder.modules()))
        # rpn4_loc_decoder_modules = list(map(id, model.rpn_head.rpn4.loc.decoder.modules()))
        # backbone_modules = list(map(id, model.backbone.modules()))
        # base_modules = filter(
        #     lambda p: id(
        #         p) not in rpn2_cls_decoder_modules + rpn2_cls_encoder_modules + rpn2_loc_decoder_modules + rpn2_loc_encoder_modules \
        #               + rpn3_cls_decoder_modules + rpn3_cls_encoder_modules + rpn3_loc_decoder_modules + rpn3_loc_encoder_modules \
        #               + rpn4_cls_decoder_modules + rpn4_cls_encoder_modules + rpn4_loc_decoder_modules + rpn4_loc_encoder_modules \
        #               + backbone_modules,
        #     model.modules())
        for m in base_modules:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        # for debug
        # for name, m in model.named_modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         if m.training:
        #             print("training: True ", name)
        #         else:
        #             print("training: False ", name)

    return model, base_params, base_modules
# def load_pretrain(model, pretrained_path):
#     logger.info('load pretrained model from {}'.format(pretrained_path))
#     device = torch.cuda.current_device()
#     pretrained_dict = torch.load(pretrained_path,
#         map_location=lambda storage, loc: storage.cuda(device))
#     if "state_dict" in pretrained_dict.keys():
#         pretrained_dict = remove_prefix(pretrained_dict['state_dict'],
#                                         'module.')
#     else:
#         pretrained_dict = remove_prefix(pretrained_dict, 'module.')
#
#     try:
#         check_keys(model, pretrained_dict)
#     except:
#         logger.info('[Warning]: using pretrain as features.\
#                 Adding "features." as prefix')
#         new_dict = {}
#         for k, v in pretrained_dict.items():
#             k = 'features.' + k
#             new_dict[k] = v
#         pretrained_dict = new_dict
#         check_keys(model, pretrained_dict)
#     model.load_state_dict(pretrained_dict, strict=False)
#     return model


def restore_from(model, optimizer, ckpt_path):
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path,
        map_location=lambda storage, loc: storage.cuda(device))
    epoch = ckpt['epoch']

    ckpt_model_dict = remove_prefix(ckpt['state_dict'], 'module.')
    check_keys(model, ckpt_model_dict)
    model.load_state_dict(ckpt_model_dict, strict=False)

    check_keys(optimizer, ckpt['optimizer'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, epoch
