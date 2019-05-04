import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel
from models.modules.loss import GANLoss, GradientPenaltyLoss, VGGContextualLoss, Blur, isotropic_gaussian_kernel


class SRGANModel(BaseModel):
    def name(self):
        return 'SRGANModel'

    def __init__(self, opt):
        super(SRGANModel, self).__init__(opt)
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)  # D
            self.netG.train()
            self.netD.train()
        self.load()  # load G and D if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [%s] is not recognized.' % l_pix_type)
                self.l_pix_w = train_opt['pixel_weight']
            else:
                print('Remove pixel loss.')
                self.cri_pix = None

            # G blur loss
            if train_opt['blur_weight'] > 0:
                l_blur_type = train_opt['blur_criterion']
                if l_blur_type == 'l1':
                    self.cri_blur = nn.L1Loss().to(self.device)
                elif l_blur_type == 'l2':
                    self.cri_blur = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [%s] is not recognized.' % l_blur_type)
                self.l_blur_w = train_opt['blur_weight']
            else:
                print('Remove blur loss.')
                self.cri_blur = None
            if self.cri_blur:  # blure loss
                blur_kernel = isotropic_gaussian_kernel(l=21, sigma=3.0)
                self.blur = Blur(l=21, kernel=blur_kernel).to(self.device)

            # G contextual loss
            if train_opt['cx_weight'] > 0:
                self.l_cx_w = train_opt['cx_weight']
                self.contextual = VGGContextualLoss(opt, use_bn=False, cxloss_type=train_opt['cxloss_type']).to(self.device)

                self.netF = networks.define_F(opt, use_bn=False).to(self.device)

            else:
                print('Remove cx loss.')

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            # 对抗函数有很多
            if train_opt['gan_type'] == 'wgan-gp':
                self.random_pt = torch.Tensor(1, 1, 1, 1).to(self.device)
                # gradient penalty loss
                self.cri_gp = GradientPenaltyLoss(device=self.device).to(self.device)
                self.l_gp_w = train_opt['gp_weigth']

            # optimizers
            self.optimizers = []  # G and D
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print('WARNING: params [%s] will not optimize.' % k)
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'], \
                weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
            self.optimizers.append(self.optimizer_D)

            # schedulers
            # 改变learning rate
            self.schedulers = []
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        print('---------- Model initialized ------------------')
        self.print_network()
        print('-----------------------------------------------')

    def feed_data(self, data, need_HR=True):
        # LR
        self.var_L = data['LR'].to(self.device)

        if need_HR:  # train or val
            self.var_H = data['HR'].to(self.device)

            input_ref = data['ref'] if 'ref' in data else data['HR']
            self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self, step):
        # G
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:

            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix

            if self.cri_blur:  # blur loss
                l_g_blur = self.l_blur_w * self.cri_blur(self.blur(self.fake_H), self.blur(self.var_H))
                l_g_total += l_g_blur

            if self.l_cx_w > 0:  # contextual loss
                l_g_cx = self.l_cx_w * self.contextual(self.fake_H, self.var_H)
                l_g_total += l_g_cx

            # G gan + cls loss
            pred_g_fake = self.netD(self.fake_H)
            # G网络就是要生成真的，对抗学习
            l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
            l_g_total += l_g_gan

            l_g_total.backward()
            self.optimizer_G.step()

        # D
        self.optimizer_D.zero_grad()
        l_d_total = 0
        # real data
        pred_d_real = self.netD(self.var_ref)
        # D网络就是要辨认真假，对抗学习
        l_d_real = self.cri_gan(pred_d_real, True)
        # fake data
        pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G，对D网络，G网络是参数
        l_d_fake = self.cri_gan(pred_d_fake, False)

        l_d_total = l_d_real + l_d_fake

        if self.opt['train']['gan_type'] == 'wgan-gp':
            batch_size = self.var_ref.size(0)
            if self.random_pt.size(0) != batch_size:
                self.random_pt.resize_(batch_size, 1, 1, 1)
            self.random_pt.uniform_()  # Draw random interpolation points
            interp = self.random_pt * self.fake_H.detach() + (1 - self.random_pt) * self.var_ref
            interp.requires_grad = True
            interp_crit, _ = self.netD(interp)
            l_d_gp = self.l_gp_w * self.cri_gp(interp, interp_crit)  # maybe wrong in cls?
            l_d_total += l_d_gp

        l_d_total.backward()
        self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            # G
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_blur:
                self.log_dict['l_g_blur'] = l_g_blur.item()
            if self.l_cx_w > 0:
                self.log_dict['l_g_cx'] = l_g_cx.item()
            self.log_dict['l_g_gan'] = l_g_gan.item()
        # D
        self.log_dict['l_d_real'] = l_d_real.item()
        self.log_dict['l_d_fake'] = l_d_fake.item()

        if self.opt['train']['gan_type'] == 'wgan-gp':
            self.log_dict['l_d_gp'] = l_d_gp.item()
        # D outputs
        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

    def test(self):
        self.netG.eval()
        self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        print('Number of parameters in G: {:,d}'.format(n))
        if self.is_train:
            message = '-------------- Generator --------------\n' + s + '\n'
            network_path = os.path.join(self.save_dir, '../', 'network.txt')
            with open(network_path, 'w') as f:
                f.write(message)

            # Discriminator
            s, n = self.get_network_description(self.netD)
            print('Number of parameters in D: {:,d}'.format(n))
            message = '\n\n\n-------------- Discriminator --------------\n' + s + '\n'
            with open(network_path, 'a') as f:
                f.write(message)

            if self.netF:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                print('Number of parameters in F: {:,d}'.format(n))
                message = '\n\n\n-------------- Feature Network --------------\n' + s + '\n'
                with open(network_path, 'a') as f:
                    f.write(message)

# 迁移学习
    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            print('loading model for G [%s] ...' % load_path_G)
            self.load_network(load_path_G, self.netG)
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            print('loading model for D [%s] ...' % load_path_D)
            self.load_network(load_path_D, self.netD)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        self.save_network(self.save_dir, self.netD, 'D', iter_label)
