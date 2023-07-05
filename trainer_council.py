"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, MsImageDisCouncil
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import random
import threading
from multiprocessing.pool import ThreadPool
import warnings
from collections import deque
import numpy as np
import torchvision.transforms.functional as TF
from scipy import ndimage
from attribute_discriminant.model import BitmojiGenderClassifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Council_Trainer(nn.Module):
    def __init__(self, hyperparameters, cuda_device='cuda:0'):
        super(Council_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.council_size = hyperparameters['council']['council_size']
        self.council_size_conf = self.council_size
        self.gen_a2b_s = []
        self.dis_a2b_s = []
        self.do_dis_council = hyperparameters['council_w'] != 0
        self.numberOfCouncil_dis_relative_iteration_conf = hyperparameters['council']['numberOfCouncil_dis_relative_iteration']
        self.discriminetro_less_style_by_conf = hyperparameters['council']['discriminetro_less_style_by']
        self.cuda_device = cuda_device

        # all varible with '_conf' at the end will be saved and displayed in tensorboard logs

        self.flipOnOff_On_iteration_conf = hyperparameters['council']['flipOnOff_On_iteration']
        self.flipOnOff_Off_iteration_conf = hyperparameters['council']['flipOnOff_Off_iteration']
        self.flipOnOff_Off_iteration_conf = hyperparameters['council']['flipOnOff_start_with']
        self.council_w_conf = hyperparameters['council_w']
        self.council_start_at_iter_conf = hyperparameters['council']['council_start_at_iter']
        self.focus_loss_start_at_iter_conf = hyperparameters['focus_loss']['focus_loss_start_at_iter']
        self.mask_zero_or_one_center_conf = hyperparameters['focus_loss']['mask_zero_or_one_center']
        self.mask_zero_or_one_epsilon_conf = hyperparameters['focus_loss']['mask_zero_or_one_epsilon']
        self.batch_size_conf = hyperparameters['batch_size']
        self.do_w_loss_matching = hyperparameters['do_w_loss_matching']
        self.do_w_loss_matching_focus = hyperparameters['focus_loss']['do_w_loss_matching_focus']
        self.los_matching_hist_size_conf = hyperparameters['loss_matching_hist_size']
        self.do_a2b_conf = hyperparameters['do_a2b']
        self.w_match_b2a_conf = 1
        self.w_match_a2b_conf = 1
        self.w_match_focus_a2b_conf = 1
        self.w_match_focus_b2a_conf = 1
        self.w_match_focus_zero_one_a2b_conf = 1
        self.w_match_focus_zero_one_b2a_conf = 1

        if self.do_a2b_conf:
            self.los_hist_gan_a2b_s = []
            self.los_hist_council_a2b_s = []
            self.los_hist_ac_gan_a2b_s = []


        for ind in range(self.council_size):
            if self.do_a2b_conf:
                self.los_hist_gan_a2b_s.append(deque(np.ones(self.los_matching_hist_size_conf)))
                self.los_hist_council_a2b_s.append(deque(np.ones(self.los_matching_hist_size_conf)))
                self.los_hist_ac_gan_a2b_s.append(deque(np.ones(self.los_matching_hist_size_conf)))


        self.do_council_loss = None

        if self.do_dis_council:
            self.dis_council_a2b_s = []

        # defining all the networks
        for i in range(self.council_size):
            if self.do_a2b_conf:
                self.gen_a2b_s.append(
                    AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'], cuda_device=self.cuda_device))  # auto-encoder for domain a2b
                self.dis_a2b_s.append(
                    MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'], cuda_device=self.cuda_device))  # discriminator for domain a2b
                if self.do_dis_council:
                    self.dis_council_a2b_s.append(
                        MsImageDisCouncil(hyperparameters['input_dim_a'],
                                          hyperparameters['dis'], cuda_device=self.cuda_device))  # council discriminator for domain a2b
        
        
        # define gender classifier
        b_gender_discriminant = BitmojiGenderClassifier().to(device)
        b_gender_discriminant.load_state_dict(torch.load(hyperparameters['bitmoji_classifier_model_weights_path']))
        for name, para in b_gender_discriminant.named_parameters():
            para.requires_grad = False
        self.b_gender_discriminant = b_gender_discriminant.eval()
        self.CE_loss = nn.CrossEntropyLoss()
        
        self.bring_ac_iter = 31000
        
            

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        if self.do_a2b_conf:
            self.gen_a2b_s = nn.ModuleList(self.gen_a2b_s)
            self.dis_a2b_s = nn.ModuleList(self.dis_a2b_s)
            if self.do_dis_council:
                self.dis_council_a2b_s = nn.ModuleList(self.dis_council_a2b_s)


        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda(self.cuda_device)
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda(self.cuda_device)
        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params_s = []
        gen_params_s = []
        self.dis_opt_s = []
        self.gen_opt_s = []
        self.dis_scheduler_s = []
        self.gen_scheduler_s = []
        if self.do_dis_council:
            dis_council_params_s = []
            self.dis_council_opt_s = []
            self.dis_council_scheduler_s = []
        for i in range(self.council_size):
            dis_parms = []
            gen_parms = []
            dis_council_parms = []
            if self.do_a2b_conf:
                dis_parms += list(self.dis_a2b_s[i].parameters())
                gen_parms += list(self.gen_a2b_s[i].parameters())
                if self.do_dis_council:
                    dis_council_parms += list(self.dis_council_a2b_s[i].parameters())

            dis_params_s.append(dis_parms)
            gen_params_s.append(gen_parms)
            if self.do_dis_council:
                dis_council_params_s.append(dis_council_parms)
            self.dis_opt_s.append(torch.optim.Adam([p for p in dis_params_s[i] if p.requires_grad],
                                                   lr=lr, betas=(beta1, beta2),
                                                   weight_decay=hyperparameters['weight_decay']))
            self.gen_opt_s.append(torch.optim.Adam([p for p in gen_params_s[i] if p.requires_grad],
                                                   lr=lr, betas=(beta1, beta2),
                                                   weight_decay=hyperparameters['weight_decay']))
            if self.do_dis_council:
                self.dis_council_opt_s.append(torch.optim.Adam([p for p in dis_council_params_s[i] if p.requires_grad],
                                                               lr=lr, betas=(beta1, beta2),
                                                               weight_decay=hyperparameters['weight_decay']))
            self.dis_scheduler_s.append(get_scheduler(self.dis_opt_s[i], hyperparameters))
            self.gen_scheduler_s.append(get_scheduler(self.gen_opt_s[i], hyperparameters))
            if self.do_dis_council:
                self.dis_council_scheduler_s.append(get_scheduler(self.dis_council_opt_s[i], hyperparameters))

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        for i in range(self.council_size):
            if self.do_a2b_conf:
                self.gen_a2b_s[i].apply(weights_init(hyperparameters['init']))
                self.dis_a2b_s[i].apply(weights_init('gaussian'))
                if self.do_dis_council:
                    self.dis_council_a2b_s[i].apply(weights_init('gaussian'))



    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def recon_criterion_v2_color(self, input, target):
        loss_1 = torch.mean(torch.abs(input - target))
        loss_2 = torch.mean(torch.pow((input - target), 2))
        if loss_1 > loss_2:
            return loss_1
        return loss_2

    def recon_criterion_v3_gray_scale(self, input, target):
        loss_1 = torch.mean(torch.abs(torch.sum(input, 1) - torch.sum(target, 1)))
        loss_2 = torch.mean(torch.pow((torch.sum(input, 1) - torch.sum(target, 1)), 2))
        if loss_1 > loss_2:
            return loss_1
        return loss_2

    def council_basic_criterion_gray_scale(self, input, target):
        return torch.mean(torch.abs(torch.sum(input, 1) - torch.sum(target, 1)))

    def council_basic_criterion_with_color(self, input, target):
        return torch.mean(torch.abs(input - target))

    def mask_zero_one_criterion(self, mask, center=0.5, epsilon=0.01):
        return torch.sum(1 / (torch.abs(mask - center) + epsilon)) / mask.numel()

    def mask_small_criterion(self, mask):
        assert self.hyperparameters['focus_loss']['mask_small_use_abs'] or self.hyperparameters['focus_loss']['mask_small_use_square'], 'at leas one small mask loss should be true, mask_small_use_abs or mask_small_use_square'
        loss = 0
        if self.hyperparameters['focus_loss']['mask_small_use_abs']:
            loss += self.mask_small_criterion_abs(mask)
        if self.hyperparameters['focus_loss']['mask_small_use_square']:
            loss += self.mask_small_criterion_square(mask)
        return loss

    def mask_small_criterion_square(self, mask):
        return (torch.sum(mask) / mask.numel()) ** 2

    def mask_small_criterion_abs(self, mask):
        return torch.abs((torch.sum(mask))) / mask.numel()

    def mask_criterion_TV(self, mask):
        return (torch.sum(torch.abs(mask[:, :, 1:, :]-mask[:, :, :-1, :])) + \
               torch.sum(torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1]))) / mask.numel()


    def forward(self, x_a, s_t=None, x_b=None, s_a=None, s_b=None):
        self.eval()
        if s_t is not None:
            s_a = s_t
            s_b = s_t
        if self.do_a2b_conf:
            s_b = self.s_b if s_b is None else s_b
            s_b = Variable(s_b)
            x_ab_s = []
        for i in range(self.council_size):
            if self.do_a2b_conf:
                c_a, s_a_fake = self.gen_a2b.encode(x_a) # TODO what the dog doin?!!
                x_ab_s.append(self.gen_a2b.decode(c_a, s_b, x_a))


        return x_ab_s

    def gen_update(self, x_a, x_b, hyperparameters, attr_a=None, attr_b=None, iterations=0):
        self.hyperparameters = hyperparameters
        for gen_opt in self.gen_opt_s:
            gen_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda(self.cuda_device))
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda(self.cuda_device))
        c_a_s = []

        x_ab_s = []

        self.loss_gen_adv_a2b_s = []
        self.loss_gender_disc_loss_a2b_s = []
        self.loss_gen_total_s = []
        self.council_w_conf = hyperparameters['council_w'] if hyperparameters['iteration'] > hyperparameters['council']['council_start_at_iter'] else 0
        for i in range(self.council_size):
            # encode
            if self.do_a2b_conf:
                c_a, s_a_prime = self.gen_a2b_s[i].encode(x_a, attr_a, hyperparameters['iteration'])
                c_a_s.append(c_a)


            # decode (cross domain)
            if self.do_a2b_conf:
                x_ab_s.append(self.gen_a2b_s[i].decode(c_a_s[i], s_b, x_a))


            self.loss_gen_total_s.append(0)


            # GAN loss
            if hyperparameters['gan_w'] != 0:
                i_dis = i

                if hyperparameters['do_a2b']:
                    x_ab_s_curr = x_ab_s[i]
                    loss_gen_adv_a2b = self.dis_a2b_s[i_dis].calc_gen_loss(x_ab_s_curr)
                    
                    # if hyperparameters['iteration'] > self.bring_ac_iter:
                    #     preds = self.b_gender_discriminant(x_ab_s_curr)
                    #     loss_gender_disc_loss_a2b = self.CE_loss(preds, attr_a)
                    # else:
                    #     loss_gender_disc_loss_a2b = 0

                else:
                    loss_gen_adv_a2b = 0
                


                self.loss_gen_adv_a2b_s.append(loss_gen_adv_a2b)
                
                # self.loss_gender_disc_loss_a2b_s.append(loss_gender_disc_loss_a2b)


                if self.do_w_loss_matching:
                    if hyperparameters['do_a2b']:
                        self.los_hist_gan_a2b_s[i].append(loss_gen_adv_a2b.detach().cpu().numpy())
                        self.los_hist_gan_a2b_s[i].popleft()
                        # if hyperparameters['iteration'] > self.bring_ac_iter:
                        #     self.los_hist_ac_gan_a2b_s[i].append(loss_gender_disc_loss_a2b.detach().cpu().numpy())
                        #     self.los_hist_ac_gan_a2b_s[i].popleft()


                if hyperparameters['do_a2b']:
                    self.loss_gen_total_s[i] += hyperparameters['gan_w'] * self.loss_gen_adv_a2b_s[i].cuda(self.cuda_device)
                    # if hyperparameters['iteration'] > self.bring_ac_iter:
                    #     self.loss_gen_total_s[i] += hyperparameters['ac_gan_w'] * self.loss_gender_disc_loss_a2b_s[i].cuda(self.cuda_device)




        # Council loss
        onOffCycle = hyperparameters['council']['flipOnOff_On_iteration'] + hyperparameters['council'][
            'flipOnOff_Off_iteration']
        currIterCyc = hyperparameters['iteration'] % onOffCycle
        if hyperparameters['council']['flipOnOff_start_with']:
            startCyc = hyperparameters['council']['flipOnOff_On_iteration']
        else:
            startCyc = hyperparameters['council']['flipOnOff_Off_iteration']

        self.do_council_loss = hyperparameters['council']['flipOnOff_start_with'] if (currIterCyc < startCyc) \
            else not hyperparameters['council']['flipOnOff_start_with']

        if not hyperparameters['council']['flipOnOff']:
            self.do_council_loss = True
        if hyperparameters['iteration'] < hyperparameters['council']['council_start_at_iter']:
            self.do_council_loss = False

        self.council_loss_ab_s = []
        for i in range(self.council_size):
            if (hyperparameters['council_w'] != 0) and self.do_council_loss and self.council_size > 1:
                # if i == 0:
                #     print('do council loss: True')
                if self.do_a2b_conf:
                    self.council_loss_ab_s.append(0)


                if self.do_dis_council:  # do council discriminator
                    if hyperparameters['do_a2b']:
                        dis_council_loss_ab = self.dis_council_a2b_s[i].calc_gen_loss(x_ab_s[i], x_a)
                    else:
                        dis_council_loss_ab = 0

                    if self.do_w_loss_matching:
                        if hyperparameters['do_a2b']:
                            self.los_hist_council_a2b_s[i].append(dis_council_loss_ab.detach().cpu().numpy())
                            self.los_hist_council_a2b_s[i].popleft()
                            self.w_match_a2b_conf = np.mean(self.los_hist_gan_a2b_s[i]) / np.mean(self.los_hist_council_a2b_s[i])
                            dis_council_loss_ab *= self.w_match_a2b_conf


                    if hyperparameters['do_a2b']:
                        dis_council_loss_ab *= hyperparameters['council_w']
                        self.council_loss_ab_s[i] += dis_council_loss_ab



                if hyperparameters['do_a2b']:
                    self.loss_gen_total_s[i] += self.council_loss_ab_s[i].cuda(self.cuda_device)


            else:
                if self.do_a2b_conf:
                    self.council_loss_ab_s.append(0)


            # backpropogation
            self.loss_gen_total_s[i].backward()
            self.gen_opt_s[i].step()


    def sample(self, x_a=None, x_b=None, attrs_a=None, attrs_b=None, s_a=None, s_b=None, council_member_to_sample_vec=None, return_mask=True, iteration=0):
        self.eval()
        if self.do_a2b_conf:
            x_a_s = []
            s_b = self.s_b if s_b is None else s_b
            s_b1 = Variable(s_b)
            s_b2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda(self.cuda_device))
            x_a_recon, x_ab1, x_ab2, x_ab1_mask = [], [], [], []


        council_member_to_sample_vec = range(self.council_size) if council_member_to_sample_vec is None else council_member_to_sample_vec
        x_size = x_a.size(0) if x_a is not None else x_b.size(0)
        for i in range(x_size):
            for j in council_member_to_sample_vec:
                if self.do_a2b_conf:
                    x_a_s.append(x_a[i].unsqueeze(0))
                    # print(x_a[i].unsqueeze(0).shape)
                    # print(attrs_a)
                    # print(attrs_a[i])

                    c_a, s_a_fake = self.gen_a2b_s[j].encode(x_a[i].unsqueeze(0), torch.tensor([attrs_a[i]], dtype=torch.int).cuda(self.cuda_device), iteration) ## TODO attention
                    if not return_mask:
                        x_a_recon.append(self.gen_a2b_s[j].decode(c_a, s_a_fake, x_a[i].unsqueeze(0)))
                        x_ab1.append(self.gen_a2b_s[j].decode(c_a, s_b1[i].unsqueeze(0), x_a[i].unsqueeze(0)))
                        x_ab2.append(self.gen_a2b_s[j].decode(c_a, s_b2[i].unsqueeze(0), x_a[i].unsqueeze(0)))
                    else:
                        x_ab1_tmp, x_ab1_mask_tmp = self.gen_a2b_s[j].decode(c_a, s_b1[i].unsqueeze(0), x_a[i].unsqueeze(0), return_mask=return_mask)
                        do_double = False
                        if do_double:
                            c_a_double, s_a_fake = self.gen_a2b_s[j].encode(x_ab1_tmp)
                            x_ab1_tmp, x_ab1_mask_tmp = self.gen_a2b_s[j].decode(c_a_double, s_b1[i].unsqueeze(0),
                                                                               x_ab1_tmp,
                                                                               return_mask=return_mask)

                        x_ab1_mask.append(x_ab1_mask_tmp)
                        x_ab1.append(x_ab1_tmp)
                        x_ab2.append(self.gen_a2b_s[j].decode(c_a, s_b2[i].unsqueeze(0), x_a[i].unsqueeze(0)))

        if self.do_a2b_conf:
            x_a_s = torch.cat(x_a_s)
            x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
            if not return_mask:
                x_a_recon = torch.cat(x_a_recon)
            else:
                x_ab1_mask = torch.cat(x_ab1_mask)

        self.train()

        do_diff = False
        if do_diff:
            if self.do_a2b_conf:
                x_ab1 = x_a_s - x_ab1
                x_ab2 = x_a_s - x_ab2


        if not return_mask:
            if self.do_a2b_conf:
                return x_a_s, x_a_recon, x_ab1, x_ab2, None, None, None, None
        else:
            if self.do_a2b_conf:
                return x_a_s, x_ab1_mask, x_ab1, x_ab2, None, None, None, None


    def dis_update(self, x_a=None, x_b=None, attr_a=None, attr_b=None, hyperparameters=None):

        x_a_dis = x_a
        x_b_dis = x_b 
        for dis_opt in self.dis_opt_s:
            dis_opt.zero_grad()
        if self.do_a2b_conf:
            s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda(self.cuda_device))
            self.loss_dis_a2b_s = []
        self.loss_dis_total_s = []
        for i in range(self.council_size):
            i_gen = i
            if hyperparameters['dis']['useRandomGen']:
                i_gen = np.random.randint(self.council_size)

            # encode
            if hyperparameters['do_a2b']:
                c_a, _ = self.gen_a2b_s[i_gen].encode(x_a, attr_a, hyperparameters['iteration'])


            # decode (cross domain)
            if hyperparameters['do_a2b']:
                x_ab = self.gen_a2b_s[i_gen].decode(c_a, s_b, x_a)
                x_ab = x_ab if not hyperparameters['dis']['do_Dis_only_gray'] else torch.sum(x_ab.detach(), 1).unsqueeze(1).repeat(1, hyperparameters['input_dim_b'], 1, 1) / hyperparameters['input_dim_b']


            # D loss
            if hyperparameters['do_a2b']:
                self.loss_dis_a2b_s.append(self.dis_a2b_s[i].calc_dis_loss(x_ab.detach(), x_b_dis))


            self.loss_dis_total_s.append(0)
            if hyperparameters['do_a2b']:
                self.loss_dis_total_s[i] += hyperparameters['gan_w'] * self.loss_dis_a2b_s[i]


            self.loss_dis_total_s[i].backward()
            self.dis_opt_s[i].step()

    def dis_council_update(self, x_a=None, x_b=None, attr_a=None, attr_b=None, hyperparameters=None):


        if self.council_size <= 1 or hyperparameters['council']['numberOfCouncil_dis_relative_iteration'] == 0:
            print('no council discriminetor is needed (council size <= 1 or numberOfCouncil_dis_relative_iteration == 0)')
            return
        onOffCycle = hyperparameters['council']['flipOnOff_On_iteration'] + hyperparameters['council'][
            'flipOnOff_Off_iteration']
        currIterCyc = hyperparameters['iteration'] % onOffCycle
        if hyperparameters['council']['flipOnOff_start_with']:
            startCyc = hyperparameters['council']['flipOnOff_On_iteration']
        else:
            startCyc = hyperparameters['council']['flipOnOff_Off_iteration']

        self.do_council_loss = hyperparameters['council']['flipOnOff_start_with'] if (currIterCyc < startCyc) \
            else not hyperparameters['council']['flipOnOff_start_with']
        if not hyperparameters['council']['flipOnOff']:
            self.do_council_loss = hyperparameters['council']['flipOnOff_start_with']

        if not self.do_council_loss or hyperparameters['council_w'] == 0 or hyperparameters['iteration'] < hyperparameters['council']['council_start_at_iter']:
            return

        for dis_council_opt in self.dis_council_opt_s:
            dis_council_opt.zero_grad()


        if self.do_a2b_conf:
            s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda(self.cuda_device))
        if hyperparameters['council']['discriminetro_less_style_by'] != 0:
            if self.do_a2b_conf:
                s_b_less = s_b * hyperparameters['council']['discriminetro_less_style_by']

        self.loss_dis_council_a2b_s = []
        self.loss_dis_council_total_s = []
        c_a_s = []
        x_ab_s = []
        x_ab_s_less = []

        for i in range(self.council_size):
            # encode
            if hyperparameters['do_a2b']:
                c_a, _ = self.gen_a2b_s[i].encode(x_a, attr_a, hyperparameters['iteration'])
                c_a_s.append(c_a)


            # decode (cross domain)
            if hyperparameters['do_a2b']:
                x_ab = self.gen_a2b_s[i].decode(c_a, s_b, x_a)
                x_ab_s.append(x_ab)


            if hyperparameters['council']['discriminetro_less_style_by'] != 0:
                # decode (cross domain) less_style_by
                if hyperparameters['do_a2b']:
                    x_ab_less = self.gen_a2b_s[i].decode(c_a, s_b_less, x_a)
                    x_ab_s_less.append(x_ab_less)



        if self.do_a2b_conf:
            comper_x_ab_s = x_ab_s if hyperparameters['council']['discriminetro_less_style_by'] == 0 else x_ab_s_less


        for i in range(self.council_size):
            self.loss_dis_council_a2b_s.append(0)
            index_to_chose_from = list(range(0, i)) + list(range(i + 1, self.council_size))
            for k in range(hyperparameters['council']['numberOfCouncil_dis_relative_iteration']):
                if k == self.council_size:
                    break
                if len(index_to_chose_from) == 0:
                    index_to_chose_from = list(range(0, i)) + list(range(i + 1, self.council_size)) # reinitilize the indexes to chose from if numberOfCouncil_dis_relative_iteration is biger then thr number of council members
                index_to_comper = random.choice(index_to_chose_from)
                index_to_chose_from.remove(index_to_comper)

                # D loss
                if hyperparameters['do_a2b']:
                    self.loss_dis_council_a2b_s[i] += self.dis_council_a2b_s[i].calc_dis_loss(x_ab_s[i].detach(), comper_x_ab_s[index_to_comper].detach(), x_a)  # original


            self.loss_dis_council_total_s.append(0)
            if hyperparameters['do_a2b']:
                self.loss_dis_council_total_s[i] += hyperparameters['council_w'] * self.loss_dis_council_a2b_s[i] / hyperparameters['council']['numberOfCouncil_dis_relative_iteration']


            self.loss_dis_council_total_s[i].backward()
            self.dis_council_opt_s[i].step()

    def update_learning_rate(self):
        for dis_scheduler in self.dis_scheduler_s:
            if dis_scheduler is not None:
                dis_scheduler.step()
        for gen_scheduler in self.gen_scheduler_s:
            if gen_scheduler is not None:
                gen_scheduler.step()
        if not self.do_dis_council:
            return
        for dis_council_scheduler in self.dis_council_scheduler_s:
            if dis_council_scheduler is not None:
                dis_council_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        iterations = 0
        # Load generators
        for i in range(self.council_size):
            last_model_name = get_model_list(checkpoint_dir, "gen_" + str(i))
            if last_model_name is not None:
                last_model_name = last_model_name.replace('a2b_gen_', 'gen_').replace('b2a_gen_', 'gen_')
                print('loading: ' + last_model_name)
                if self.do_a2b_conf:
                    state_dict = torch.load(last_model_name.replace('gen_', 'a2b_gen_'), map_location=torch.device(self.cuda_device))
                    self.gen_a2b_s[i].load_state_dict(state_dict['a2b'])

                iterations = int(last_model_name[-11:-3])
            else:
                warnings.warn('Failed to find gen checkpoint, did not load model')

            # Load discriminators
            last_model_name = get_model_list(checkpoint_dir, "dis_" + str(i))
            if last_model_name is not None:
                last_model_name = last_model_name.replace('a2b_dis_', 'dis_').replace('b2a_dis_', 'dis_')
                print('loading: ' + last_model_name)
                if self.do_a2b_conf:
                    state_dict = torch.load(last_model_name.replace('dis_', 'a2b_dis_'), map_location=torch.device(self.cuda_device))
                    self.dis_a2b_s[i].load_state_dict(state_dict['a2b'])

            else:
                warnings.warn('Failed to find dis checkpoint, did not load model')
            # Load council discriminators
            if self.do_dis_council:
                try:
                    last_model_name = get_model_list(checkpoint_dir, "dis_council_" + str(i))
                    print('loading: ' + last_model_name)
                    if last_model_name is not None:
                        last_model_name = last_model_name.replace('a2b_dis_council_', 'dis_council_').replace('b2a_dis_council_', 'dis_council_')

                        if self.do_a2b_conf:
                            state_dict = torch.load(last_model_name.replace('dis_council_', 'a2b_dis_council_'), map_location=torch.device(self.cuda_device))
                            self.dis_council_a2b_s[i].load_state_dict(state_dict['a2b'])

                    else:
                        warnings.warn('Failed to find dis checkpoint, did not load model')
                except:
                    warnings.warn('some council discriminetor FAILED to load')

            # Load optimizers
            try:
                state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer_' + str(i) + '.pt'), map_location=torch.device(self.cuda_device))
                self.dis_opt_s[i].load_state_dict(state_dict['dis'])
                self.gen_opt_s[i].load_state_dict(state_dict['gen'])
                if self.do_dis_council:
                    self.dis_council_opt_s[i].load_state_dict(state_dict['dis_council'])

                # Reinitilize schedulers
                self.dis_scheduler_s[i] = get_scheduler(self.dis_opt_s[i], hyperparameters, iterations)
                self.gen_scheduler = get_scheduler(self.gen_opt_s[i], hyperparameters, iterations)
                if self.do_dis_council:
                    self.dis_council_scheduler_s[i] = get_scheduler(self.dis_council_opt_s[i], hyperparameters, iterations)
            except:
                warnings.warn('some optimizer FAILED to load ')
        if iterations > 0 :
            print('Resume from iteration %d' % iterations)
        else:
            warnings.warn('FAILED TO RESUME STARTED FROM 0')
        return iterations

    def save(self, snapshot_dir, iterations):
        for i in range(self.council_size):

            # Save generators, discriminators, and optimizers
            gen_name = os.path.join(snapshot_dir, 'gen_' + str(i) + '_%08d.pt' % (iterations + 1))
            dis_name = os.path.join(snapshot_dir, 'dis_' + str(i) + '_%08d.pt' % (iterations + 1))
            if self.do_dis_council:
                dis_council_name = os.path.join(snapshot_dir, 'dis_council_' + str(i) + '_%08d.pt' % (iterations + 1))
            opt_name = os.path.join(snapshot_dir, 'optimizer_' + str(i) + '.pt')
            if self.do_a2b_conf:
                torch.save({'a2b': self.gen_a2b_s[i].state_dict()}, gen_name.replace('gen_', 'a2b_gen_'))
                torch.save({'a2b': self.dis_a2b_s[i].state_dict()}, dis_name.replace('dis_', 'a2b_dis_'))

            if self.do_dis_council:
                if self.do_a2b_conf:
                    torch.save({'a2b': self.dis_council_a2b_s[i].state_dict()}, dis_council_name.replace('dis_council_', 'a2b_dis_council_'))

                torch.save({'gen': self.gen_opt_s[i].state_dict(), 'dis': self.dis_opt_s[i].state_dict(),
                            'dis_council': self.dis_council_opt_s[i].state_dict()}, opt_name)
            else:
                torch.save({'gen': self.gen_opt_s[i].state_dict(), 'dis': self.dis_opt_s[i].state_dict()}, opt_name)

#%%
