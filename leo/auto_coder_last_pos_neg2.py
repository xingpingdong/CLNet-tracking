from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.distributions as tdist
from pysot.core.config import cfg

class Coder(nn.Module):
    def __init__(self):
        super(Coder, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class EncoderCls(Coder):
    def __init__(self, in_channels, n_latent, is_meta_training=True):
        super(EncoderCls, self).__init__()
        self.in_channels = in_channels
        self.is_meta_training = is_meta_training
        self.n_latent = n_latent
        print('cfg.LATENTS.ENCODER_ADJUST', cfg.LATENTS.ENCODER_ADJUST)
        if cfg.LATENTS.ENCODER_ADJUST == '3-layer':
            self.encoder_mu = Adjust(in_channels, n_latent)
        elif cfg.LATENTS.ENCODER_ADJUST == '0-layer':
            self.encoder_mu = Adjust0(in_channels, n_latent)
        elif cfg.LATENTS.ENCODER_ADJUST == '1-layer':
            self.encoder_mu = Adjust1(in_channels, n_latent)

        self.poss_sample = PossiblySample()

    def get_positive_map(self, label_cls):
        pos_map = label_cls.data.eq(1)
        pos_map = torch.sum(pos_map,1).ge(1)
        neg_map = label_cls.data.ge(0)
        neg_map = torch.sum(neg_map,1).ge(1)
        neg_map = neg_map - pos_map

        pos = pos_map.view(-1).nonzero().squeeze().cuda()
        neg = neg_map.view(-1).nonzero().squeeze().cuda()

        return pos, neg


    def forward(self, hidden_map, label_cls):
        mus = self.encoder_mu(hidden_map)
        pos, neg = self.get_positive_map(label_cls)
        mu, s1 = self.average_codes_per_class(mus, pos, neg)
        kl = 0.
        sigma, _ = self.std_codes_per_class(mus, pos, neg)
        mu = torch.cat((mu, sigma), 1)
        latents = mu.view(mu.shape[0], mu.shape[1], mu.shape[2], 1)

        return latents, kl, s1

    def average_codes_per_class(self, codes, pos, neg):
        s = codes.shape
        codes = codes.permute(1, 0, 2, 3).contiguous().view(s[1], -1)

        if bool(pos.numel()):
            codes_pos = torch.index_select(codes, 1, pos)
            codes_pos = torch.mean(codes_pos, dim=1).view(1, -1, 1, 1)
        else:
            codes_pos = torch.zeros(s[1]).cuda().view(1, s[1], 1, 1)
            if torch.isnan(codes_pos).sum():
                codes_pos[torch.isnan(codes_pos)] = 0
        codes_neg = codes.index_select(1, neg)
        codes_neg = torch.mean(codes_neg, dim=1).view(1, -1, 1, 1)
        codes = torch.cat((codes_pos, codes_neg), 1)

        return codes, s

    def std_codes_per_class(self, codes, pos, neg):
        s = codes.shape
        codes = codes.permute(1, 0, 2, 3).contiguous().view(s[1], -1)
        if pos.numel()>1:
            codes_pos = torch.index_select(codes, 1, pos)
            codes_pos = torch.std(codes_pos, dim=1).view(1, -1, 1, 1)
        else:
            codes_pos = torch.zeros(s[1]).cuda().view(1, s[1], 1, 1)
            if torch.isnan(codes_pos).sum():
                codes_pos[torch.isnan(codes_pos)] = 0
        codes_neg = codes.index_select(1, neg)
        codes_neg = torch.std(codes_neg, dim=1).view(1, -1, 1, 1)
        codes = torch.cat((codes_pos, codes_neg), 1)

        if torch.isnan(codes).sum():
            print(torch.isnan(codes).sum())
            # print(codes_neg0)
        return codes, s

class EncoderLoc(Coder):
    def __init__(self, in_channels, n_latent, is_meta_training=True):
        super(EncoderLoc, self).__init__()
        self.in_channels = in_channels
        self.is_meta_training = is_meta_training
        self.n_latent = n_latent
        if cfg.LATENTS.ENCODER_ADJUST == '3-layer':
            self.encoder_mu = Adjust(in_channels, n_latent)
        elif cfg.LATENTS.ENCODER_ADJUST == '0-layer':
            self.encoder_mu = Adjust0(in_channels, n_latent)
        elif cfg.LATENTS.ENCODER_ADJUST == '1-layer':
            self.encoder_mu = Adjust1(in_channels, n_latent)

        self.poss_sample = PossiblySample()

    def get_positive_map(self, label_cls):
        pos_map = label_cls.data.eq(1)
        pos_map = torch.sum(pos_map,1).ge(1)
        pos = pos_map.view(-1).nonzero().squeeze().cuda()

        return pos


    def forward(self, hidden_map, label_cls):
        mus = self.encoder_mu(hidden_map)
        s = [mus.shape[2], mus.shape[3]]
        pos= self.get_positive_map(label_cls)
        mu, s1 = self.average_codes_per_class(mus, pos)
        kl = 0.
        sigma, _ = self.std_codes_per_class(mus, pos)

        mu = torch.cat((mu,sigma), 1)
        latents = mu.view(mu.shape[0], mu.shape[1], mu.shape[2], 1)
        return latents, kl, s1

    def average_codes_per_class(self, codes, pos):
        s = codes.shape
        codes = codes.permute(1, 0, 2, 3).contiguous().view(s[1], -1)

        if bool(pos.numel()):
            codes_pos = torch.index_select(codes, 1, pos)
            codes_pos = torch.mean(codes_pos, dim=1).view(1, -1, 1, 1)
        else:
            codes_pos = torch.zeros(s[1]).cuda().view(1, s[1], 1, 1)
            if torch.isnan(codes_pos).sum():
                codes_pos[torch.isnan(codes_pos)] = 0
        codes = codes_pos

        return codes, s

    def std_codes_per_class(self, codes, pos):
        s = codes.shape
        codes = codes.permute(1, 0, 2, 3).contiguous().view(s[1], -1)
        if pos.numel()>1:
            codes_pos = torch.index_select(codes, 1, pos)
            codes_pos = torch.std(codes_pos, dim=1).view(1, -1, 1, 1)
        else:
            codes_pos = torch.zeros(s[1]).cuda().view(1, s[1], 1, 1)
            if torch.isnan(codes_pos).sum():
                codes_pos[torch.isnan(codes_pos)] = 0
        codes = codes_pos

        if torch.isnan(codes).sum():
            print(torch.isnan(codes).sum())
            # print(codes_neg0)
        return codes, s

class EncoderLinear(Coder):
    def __init__(self, in_channels, n_latent, is_meta_training=True):
        super(EncoderLinear, self).__init__()
        self.in_channels = in_channels
        self.is_meta_training = is_meta_training
        self.n_latent = n_latent
        if cfg.LATENTS.ENCODER_ADJUST == '3-layer':
            self.encoder_mu = Adjust(in_channels, n_latent)
        elif cfg.LATENTS.ENCODER_ADJUST == '0-layer':
            self.encoder_mu = Adjust0(in_channels, n_latent)
        elif cfg.LATENTS.ENCODER_ADJUST == '1-layer':
            self.encoder_mu = Adjust1(in_channels, n_latent)

        self.poss_sample = PossiblySample()

    def forward(self, hidden_map, label_cls):
        mus = self.encoder_mu(hidden_map)
        s1 = mus.shape
        kl = 0.
        latents = mus

        return latents, kl, s1

class Decoder(Coder):
    def __init__(self, n_latent, de_hidden, hidden, out_channels, bias=False, is_meta_training=True):
        super(Decoder, self).__init__()
        if cfg.LATENTS.DECODER_NAME == 'deep':
            self.decoder_mu = WeightDistDeep(n_latent, de_hidden, hidden, out_channels, bias)
        elif cfg.LATENTS.DECODER_NAME == 'linear':
            self.decoder_mu = WeightDistLinear(n_latent*625 // 4, de_hidden, hidden, out_channels, bias)
        else:
            self.decoder_mu = WeightDist(n_latent, de_hidden, hidden, out_channels, bias)

        self.is_meta_training = is_meta_training
        self.hidden = hidden
        self.out_channels = out_channels

    def forward(self, latents, s1=0):
        weights_mu = self.decoder_mu(latents)
        weights_samples = weights_mu
        w_b = torch.mean(weights_samples,dim=0).squeeze()
        layer_weight = w_b[0].view(-1)
        w_b = w_b[1:]
        weights_samples = w_b.view(self.out_channels,self.hidden+1,1,1)
        weights = weights_samples[:,1:,:,:]

        bias = weights_samples[:,0,:,:]
        return weights, bias.squeeze(), layer_weight

class Adjust(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Adjust, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv1x1_3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature):
        codes = self.conv1x1(feature)
        codes = self.conv1x1_2(codes)
        codes = self.conv1x1_3(codes)

        return codes

class Adjust0(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Adjust0, self).__init__()

    def forward(self, feature):
        codes = feature
        return codes

class Adjust1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Adjust1, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature):
        codes = self.conv1x1(feature)
        return codes

class PossiblySample(nn.Module):
    def __init__(self,h=1,w=1, is_meta_training=True,stddev_offset=0.):
        super(PossiblySample, self).__init__()
        self.h = h
        self.w = w
        self.stddev_offset = stddev_offset
        self.is_meta_training = is_meta_training
        self.l1_loss = nn.L1Loss()

    def forward(self, mu, sigma, n_samples=1):
        stddev = torch.exp(sigma)
        stddev = stddev-(1. - self.stddev_offset)
        stddev = torch.clamp(stddev, 1e-10)
        distribution = tdist.Normal(mu, stddev)
        samples = distribution.rsample()
        kl_divergence = 0.0
        if self.is_meta_training:
            kl_divergence = self.kl_divergence(samples, distribution)
            if torch.isnan(kl_divergence).sum():
                print( torch.isnan(mu).sum())
                print(torch.isnan(sigma).sum())

        return samples, kl_divergence

    def l1_losses(self,mu,stddev):
        loc = torch.zeros_like(mu)
        scale = torch.ones_like(stddev)
        l1 = self.l1_loss(mu,loc)
        l2 = self.l1_loss(stddev,scale)
        return l1+l2

    def kl_divergence(self,samples,normal_distribution):
        random_prior = tdist.Normal(
            loc=torch.zeros_like(samples), scale=torch.ones_like(samples))

        input = normal_distribution.log_prob(samples)
        target = random_prior.log_prob(samples)
        kl = torch.mean(input-target)

        return kl



class WeightDist(nn.Module):
    def __init__(self, in_channels, de_hidden, hidden, out_channels,
                 bias = False, kernel_size=1):
        super(WeightDist,self).__init__()
        self.deconv1 = nn.Sequential(
            nn.Conv2d(in_channels, de_hidden, kernel_size=1, bias=bias),
            # nn.BatchNorm2d(de_hidden),
            nn.ReLU(inplace=True),
        )
        self.deconv1_2 = nn.Sequential(
            nn.Conv2d(de_hidden, de_hidden, kernel_size=1, bias=bias),
            # nn.BatchNorm2d(de_hidden),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(de_hidden, (hidden+1)*out_channels+1, kernel_size=1, bias=bias),
            # nn.BatchNorm2d((hidden+1)*out_channels),
            # nn.ReLU(inplace=True),
            nn.Tanh()
        )

    def forward(self, latents):
        hidden_layer = self.deconv1(latents)
        hidden_layer = self.deconv1_2(hidden_layer)
        weight_dist_param = self.deconv2(hidden_layer)
        return weight_dist_param


class WeightDistLinear(nn.Module):
    def __init__(self, in_channels, de_hidden, hidden, out_channels,
                 bias=False, kernel_size=1):
        super(WeightDistLinear, self).__init__()
        self.deconv1 = nn.Sequential(
            nn.Linear(in_channels, de_hidden),
            # nn.BatchNorm2d(de_hidden),
            nn.ReLU(inplace=True),
        )
        self.deconv1_2 = nn.Sequential(
            nn.Linear(de_hidden, de_hidden),
            # nn.BatchNorm2d(de_hidden),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.Linear(de_hidden, (hidden + 1) * out_channels + 1),
            # nn.BatchNorm2d((hidden+1)*out_channels),
            # nn.ReLU(inplace=True),
            nn.Tanh()
        )


    def forward(self, latents):
        latents = torch.flatten(latents, 1)
        hidden_layer = self.deconv1(latents)
        hidden_layer = self.deconv1_2(hidden_layer)
        weight_dist_param = self.deconv2(hidden_layer)
        return weight_dist_param

class WeightDistDeep(nn.Module):
    def __init__(self, in_channels, de_hidden, hidden, out_channels,
                 bias = False, kernel_size=1):
        super(WeightDistDeep,self).__init__()
        self.deconv1 = nn.Sequential(
            nn.Conv2d(in_channels, de_hidden, kernel_size=1, bias=bias),
            # nn.BatchNorm2d(de_hidden),
            nn.ReLU(inplace=True),
        )
        self.deconv1_2 = nn.Sequential(
            nn.Conv2d(de_hidden, de_hidden, kernel_size=1, bias=bias),
            # nn.BatchNorm2d(de_hidden),
            nn.ReLU(inplace=True),
        )
        self.deconv1_3 = nn.Sequential(
            nn.Conv2d(de_hidden, de_hidden, kernel_size=1, bias=bias),
            # nn.BatchNorm2d(de_hidden),
            nn.ReLU(inplace=True),
        )
        self.deconv1_4 = nn.Sequential(
            nn.Conv2d(de_hidden, de_hidden, kernel_size=1, bias=bias),
            # nn.BatchNorm2d(de_hidden),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(de_hidden, (hidden+1)*out_channels+1, kernel_size=1, bias=bias),
            # nn.BatchNorm2d((hidden+1)*out_channels),
            # nn.ReLU(inplace=True),
            nn.Tanh()
        )

    def forward(self, latents):
        hidden_layer = self.deconv1(latents)
        hidden_layer = self.deconv1_2(hidden_layer)
        hidden_layer = self.deconv1_3(hidden_layer)
        hidden_layer = self.deconv1_4(hidden_layer)
        weight_dist_param = self.deconv2(hidden_layer)
        return weight_dist_param
