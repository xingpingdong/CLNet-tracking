from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, DistModule, reduce_gradients, \
    average_reduce, get_rank, get_world_size
from pysot.utils.model_load import restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import describe, commit
from leo.model_builder_last_pos_neg2 import ModelBuilder
from pysot.datasets.dataset_in_video import TrkDataset
from pysot.core.config import cfg
from leo.model_load import load_pretrain_rpn_last, freeze_pretrain_head, get_coder_layers, get_head_layers
import torch.nn.functional as F

import cv2

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='clnet+siamrpn tracking')
parser.add_argument('--cfg', type=str,
                    default='/home/xd1/pysot-master/experiments/latent_last2/config2.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
parser.add_argument('--base_lr', type=float, default=0.005,
                    help='base learning rate')
parser.add_argument('--aug_lambda', type=float, default=0.1,
                    help='augmentation lambda')
parser.add_argument('--thr_high', type=float, default=0.6,
                    help='overlarp threshold high')
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_loader():
    logger.info("build train dataset")
    # train_dataset
    train_dataset = TrkDataset()
    logger.info("build dataset done")

    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True,
                              sampler=train_sampler)

    return train_loader

def build_opt_lr_latent_last(model, current_epoch=0):


    trainable_params = []
    # trainable_params += [{'params': filter(lambda x: x.requires_grad,
    #                                        model.backbone.parameters()),
    #                       'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.rpn_head.parameters()),
                          'lr': cfg.TRAIN.BASE_LR}]

    if cfg.MASK.MASK:
        trainable_params += [{'params': model.mask_head.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    if cfg.REFINE.REFINE:
        trainable_params += [{'params': model.refine_head.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler

def get_layers_in_base_model(model, type):
    if type == 'MultiLatentRPN':
        fixed_params = []
        fixed_modules = []
        for name, param in model.named_parameters():
            if 'encoder' in name or 'decoder' in name:
                a=0#print(name)
            elif 'last_weights0' in name or 'last_bias0' in name:
                a=0#print(name)
            else:
                fixed_params.append(param)
        for name, m in model.named_modules():
            if 'encoder' in name or 'decoder' in name:
                a=0#print(name)
            elif isinstance(m, nn.BatchNorm2d):
                fixed_modules.append(m)
        return fixed_params, fixed_modules

def set_fixed_params(fixed_params, fixed_modules, is_fixed):
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

def print_params_name(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("requires_grad: True ", name)
        else:
            print("requires_grad: False ", name)
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            if m.training:
                print("training: True ", name)
            else:
                print("training: False ", name)

def split_data(data):
    val_data = {}

    train_data_list = []
    n_batch = data['template'].shape[0]
    n_val = n_batch - cfg.LATENTS.NUM_UPDATES * cfg.LATENTS.UPDATE_BATCH_SIZE
    n_ub = cfg.LATENTS.UPDATE_BATCH_SIZE
    for k, v in data.items():
        if k == 'label_loc':
            val_data[k] = v[:n_val, :, :, :, :]
        elif k == 'bbox':
            val_data[k] = v[:n_val, :]
        else:
            val_data[k] = v[:n_val, :, :, :]
    for i in range(cfg.LATENTS.NUM_UPDATES):
        train_data = {}
        for k, v in data.items():
            if k == 'label_loc':
                train_data[k] = v[n_val+i*n_ub:n_val+(i+1)*n_ub, :, :, :, :]
            elif k == 'bbox':
                train_data[k] = v[n_val+i*n_ub:n_val+(i+1)*n_ub, :]
            else:
                train_data[k] = v[n_val+i*n_ub:n_val+(i+1)*n_ub, :, :, :]
        train_data_list.append(train_data)
    return val_data, train_data_list

def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, rpn_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            rpn_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/' + k.replace('.', '/'),
                             _norm, tb_index)
        tb_writer.add_scalar('weight_all/' + k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('w-g/' + k.replace('.', '/'),
                             w_norm / (1e-20 + _norm), tb_index)
    tot_norm = feature_norm + rpn_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    rpn_norm = rpn_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/rpn', rpn_norm, tb_index)

def log_fea_norm(feas, tb_writer, tb_index, rank):
    fea_cls = feas['cls']
    fea_loc = feas['loc']
    for i in range(len(fea_cls)):
        fea = fea_cls[i].detach()
        tb_writer.add_scalar('cls/layer{}_fea_norm'.format(i + 2), fea.norm(2), tb_index)
        # tb_writer.add_histogram('features_cls{}'.format(i),fea,tb_index)

        fea = fea_loc[i].detach()
        tb_writer.add_scalar('loc/layer{}_fea_norm'.format(i + 2), fea.norm(2), tb_index)

def Normalize(data):
    mx = data.max()
    mn = data.min()
    return (data - mn)/(mx - mn) * 255

def process_img(img):
    img = Normalize(img).astype(np.uint8).squeeze()
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

def log_fea_maps(feas, tb_writer, tb_index, rank):
    fea_cls = feas['cls']
    fea_loc = feas['loc']
    for i in range(len(fea_cls)):
        fea = fea_cls[i].detach()
        s = fea.shape
        fea = fea.norm(dim=1)
        fea = fea.cpu().numpy()
        heat_maps = []
        for j in range(s[0]):
            f0 = fea[j,:,:]
            h0 = process_img(f0)
            heat_maps.append(h0)
        heat_maps = np.stack(heat_maps)
        tb_writer.add_images('cls/images{}'.format(i), heat_maps, tb_index, dataformats='NHWC')

        # m = fea.max()
        # fea = (fea/m).view(s[0],s[2],s[3],1).repeat(1,1,1,3)
        # tb_writer.add_images('cls/images{}'.format(i), fea, tb_index, dataformats='NHWC')

        fea = fea_loc[i].detach()
        s = fea.shape
        fea = fea.norm(dim=1)
        fea = fea.cpu().numpy()
        heat_maps = []
        for j in range(s[0]):
            f0 = fea[j, :, :]
            h0 = process_img(f0)
            heat_maps.append(h0)
        heat_maps = np.stack(heat_maps)
        tb_writer.add_images('loc/images{}'.format(i), heat_maps, tb_index, dataformats='NHWC')

        # m = fea.max()
        # fea = (fea/m).view(s[0],s[2],s[3],1).repeat(1,1,1,3)
        # tb_writer.add_images('loc/images{}'.format(i), fea, tb_index, dataformats='NHWC')

def log_last_weights(model, tb_writer, tb_index, rank):

    with torch.no_grad():
        for idx in range(3):
            rpn = getattr(model.rpn_head, 'rpn' + str(idx + 2))
            w = rpn.cls.last_weights
            w0 = (w - rpn.cls.last_weights0)#.view(-1).data.cpu().numpy()
            b = rpn.cls.last_bias
            b0 = (b - rpn.cls.last_bias0)#.view(-1).data.cpu().numpy()
            tb_writer.add_scalar('cls/layer{}_w_norm'.format(idx+2), w0.norm(2), tb_index)
            tb_writer.add_scalar('cls/layer{}_b_norm'.format(idx+2), b0.norm(2), tb_index)
            # tb_writer.add_histogram('cls/d_weight{}'.format(idx), w0, tb_index)
            # tb_writer.add_histogram('cls/d_bias{}'.format(idx), b0, tb_index)

            w = rpn.loc.last_weights
            w0 = (w - rpn.loc.last_weights0)#.cpu().numpy()
            b = rpn.loc.last_bias
            b0 = (b - rpn.loc.last_bias0)#.cpu().numpy()
            tb_writer.add_scalar('loc/layer{}_w_norm'.format(idx + 2), w0.norm(2), tb_index)
            tb_writer.add_scalar('loc/layer{}_b_norm'.format(idx + 2), b0.norm(2), tb_index)
            #
            # tb_writer.add_histogram('loc/d_weight{}'.format(idx), w0, tb_index)
            # tb_writer.add_histogram('loc/d_bias{}'.format(idx), b0, tb_index)

def log_layer_weights(model, tb_writer, tb_index, rank):
    with torch.no_grad():
        for idx in range(3):
            rpn = getattr(model.rpn_head, 'rpn' + str(idx + 2))
            lw = rpn.cls.layer_weight
            tb_writer.add_scalar('cls/layer{}_weight'.format(idx + 2), lw, tb_index)
            lw = rpn.loc.layer_weight
            tb_writer.add_scalar('loc/layer{}_weight'.format(idx + 2), lw, tb_index)

def train(train_loader, model, optimizer, lr_scheduler, tb_writer):
    cur_lr = lr_scheduler.get_cur_lr()
    rank = get_rank()

    average_meter = AverageMeter()

    def is_valid_number(x):
        return not (math.isnan(x) or math.isinf(x) or x > 1e4)

    world_size = get_world_size()
    num_per_epoch = len(train_loader.dataset) // \
                    cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)
    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and \
            get_rank() == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    logger.info("model\n{}".format(describe(model.module)))
    end = time.time()

    for idx, data in enumerate(train_loader):
        if epoch != idx // num_per_epoch + start_epoch:
            epoch = idx // num_per_epoch + start_epoch

            if get_rank() == 0:
                torch.save(
                    {'epoch': epoch,
                     'state_dict': model.module.state_dict(),
                     'optimizer': optimizer.state_dict()},
                    cfg.TRAIN.SNAPSHOT_DIR + '/checkpoint_e%d.pth' % (epoch))

            if epoch == cfg.TRAIN.EPOCH:
                return

            if cfg.BACKBONE.TRAIN_EPOCH == epoch:
                logger.info('start training backbone.')
                optimizer, lr_scheduler = build_opt_lr_latent_last(model.module, epoch)
                logger.info("model\n{}".format(describe(model.module)))


            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_cur_lr()
            logger.info('epoch: {}'.format(epoch + 1))

        tb_idx = idx
        if idx % num_per_epoch == 0 and idx != 0:
            for idx, pg in enumerate(optimizer.param_groups):
                logger.info('epoch {} lr {}'.format(epoch + 1, pg['lr']))
                if rank == 0:
                    tb_writer.add_scalar('lr/group{}'.format(idx + 1),
                                         pg['lr'], tb_idx)

        data_time = average_reduce(time.time() - end)
        if rank == 0:
            tb_writer.add_scalar('time/data', data_time, tb_idx)

        outputs, feas = model(data)

        loss = outputs['out_loop_loss']

        if is_valid_number(loss.data.item()):
            optimizer.zero_grad()
            loss.backward()
            reduce_gradients(model)

            if rank == 0 and cfg.TRAIN.LOG_GRADS:
                log_grads(model.module, tb_writer, tb_idx)

            # clip gradient
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            optimizer.step()

        batch_time = time.time() - end
        batch_info = {}
        batch_info['batch_time'] = average_reduce(batch_time)
        batch_info['data_time'] = average_reduce(data_time)
        for k, v in outputs.items():
            batch_info[k] = average_reduce(v.data.item())

        average_meter.update(**batch_info)

        if rank == 0:
            if cfg.TRAIN.LOG_GRADS:
                log_last_weights(model.module, tb_writer, tb_idx, rank)
                log_fea_norm(feas, tb_writer, tb_idx, rank)
                log_layer_weights(model.module, tb_writer, tb_idx, rank)

            for k, v in batch_info.items():
                tb_writer.add_scalar(k, v, tb_idx)

            if (idx + 1) % cfg.TRAIN.PRINT_FREQ == 0:
                info = "Epoch: [{}][{}/{}] lr: {:.6f} \n".format(
                    epoch + 1, (idx + 1) % num_per_epoch,
                    num_per_epoch, cur_lr)
                for cc, (k, v) in enumerate(batch_info.items()):
                    if cc % 2 == 0:
                        info += ("\t{:s}\t").format(
                            getattr(average_meter, k))
                    else:
                        info += ("{:s}\n").format(
                            getattr(average_meter, k))
                logger.info(info)
                print_speed(idx + 1 + start_epoch * num_per_epoch,
                            average_meter.batch_time.avg,
                            cfg.TRAIN.EPOCH * num_per_epoch)
            # if (idx + 0) % 500 == 0:
            #     log_fea_maps(feas, tb_writer, tb_idx, rank)
        end = time.time()


def main():
    rank, world_size = dist_init()
    logger.info("init done")

    # load cfg
    cfg.merge_from_file(args.cfg)
    if cfg.LATENTS.HP:
        cfg.TRAIN.BASE_LR = args.base_lr
        print('base_lr = {}'.format(cfg.TRAIN.BASE_LR))
        cfg.LATENTS.AUGMENT_LAMBDA = args.aug_lambda
        print('aug_lambda = {}'.format(cfg.LATENTS.AUGMENT_LAMBDA ))
        cfg.TRAIN.THR_HIGH = args.thr_high
        print('threshold high: {}'.format(cfg.TRAIN.THR_HIGH))
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global',
                             os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                             logging.INFO)

        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    # create model
    model = ModelBuilder().cuda().train()
    dist_model = DistModule(model)

    # load pretrained backbone weights
    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cur_path, '../', cfg.BACKBONE.PRETRAINED)
        load_pretrain_rpn_last(model, backbone_path)

    # fix the SiamRPN base model
    fixed_params, fixed_modules = get_layers_in_base_model(dist_model.module, cfg.RPN.TYPE)
    set_fixed_params(fixed_params, fixed_modules, True)
    # print_params_name(dist_model.module)
    # create tensorboard writer
    if rank == 0 and cfg.TRAIN.LOG_DIR:
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    else:
        tb_writer = None

    # build optimizer and lr_scheduler
    optimizer, lr_scheduler = build_opt_lr_latent_last(dist_model.module,
                                                  cfg.TRAIN.START_EPOCH)

    # build dataset loader
    train_loader = build_data_loader()

    # resume training
    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optimizer, cfg.TRAIN.RESUME)
        dist_model = DistModule(model)
        # build optimizer and lr_scheduler
        optimizer, lr_scheduler = build_opt_lr_latent_last(dist_model.module,
                                                      cfg.TRAIN.START_EPOCH)

    logger.info(lr_scheduler)
    logger.info("model prepare done")

    # start training
    train(train_loader, dist_model, optimizer, lr_scheduler, tb_writer)


if __name__ == '__main__':
    seed_torch(args.seed)
    main()
