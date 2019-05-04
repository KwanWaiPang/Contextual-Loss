import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel
from models.modules.loss import  VGGContextualLoss


class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.load()

        if self.is_train:
            self.netG.train()

            # loss
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

            if train_opt['cx_weight'] > 0:
                self.l_cx_w = train_opt['cx_weight']
                self.contextual = VGGContextualLoss(opt, use_bn=False, cxloss_type=train_opt['cxloss_type']).to(self.device)
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)

            else:
                print('Remove cx loss.')

            # optimizers
            self.optimizers = []
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print('WARNING: params [%s] will not optimize.' % k)
            self.optimizer_G = torch.optim.Adam(optim_params,
                                                lr=train_opt['lr_G'], weight_decay=wd_G)
            self.optimizers.append(self.optimizer_G)

            # schedulers
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
        self.var_L = data['LR'].to(self.device) # LR

        if need_HR:
            self.real_H = data['HR'].to(self.device) # HR

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)
        
        l_g_total = 0

        if self.cri_pix:  # pixel loss
            l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
            l_g_total += l_g_pix

        if self.l_cx_w > 0:  # contextual loss
            l_g_cx = self.l_cx_w * self.contextual(self.real_H, self.fake_H)
            l_g_total += l_g_cx        

        l_g_total.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_g_total'] = l_g_total.item()

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
            out_dict['HR'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        print('Number of parameters in G: {:,d}'.format(n))
        if self.is_train:
            message = '-------------- Generator --------------\n' + s + '\n'
            network_path = os.path.join(self.save_dir, '../', 'network.txt')
            with open(network_path, 'w') as f:
                f.write(message)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            print('loading model for G [%s] ...' % load_path_G)
            self.load_network(load_path_G, self.netG)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
