from pgd import PGD
import torch.nn as nn
import torch
import math
import random
import numpy as np
from torch.autograd import grad


class FAT(PGD):
    def __init__(self, epsilon: float = 8/255,
                 alpha: float = 2 / 255,
                 init_max: int = 1,
                 iter_max: int = 10,
                 device: float = 'cuda',
                 is_nontarget: bool = True,
                 ) -> None:

        self.epsilon = epsilon
        self.alpha = alpha
        self.device = device
        self.iter_max = iter_max
        self.init_max = init_max
        self.beta = 0.087
        if is_nontarget:
            self.mode = 1
        else:
            self.mode = -1

    def make_var(self, outputs: dict, labels: torch.Tensor):
        '''
        outputs : {(layer, mid_output)}
        '''
        if len(labels.shape) == 1:
            pass
        elif len(labels.shape) >= 2:
            labels.squeeze_()

        '''
        Get the index of each label in a batch.
        '''
        label_list = tuple(labels.tolist())
        label_idx_list = [torch.nonzero(
            labels == label).squeeze().tolist() for label in label_list]

        loss = []
        layer_list = ['layer4.0', 'layer4.1']
        '''
        Select target and compute var loss.
        '''
        for layer, output in outputs.items():
            output: torch.Tensor
            if layer in layer_list:
                for label_idx in label_idx_list:
                    this_label_output = output[label_idx]
                    loss.append(torch.var(this_label_output.view(
                        this_label_output.shape[0], -1), dim=0).norm())
            else:
                pass

        loss = sum(loss)/len(loss)
        return loss

    def impl_loss(self, model: nn.Module, images: torch.Tensor, targets: torch.Tensor, loss_func: torch.Tensor):
        outputs = model.forward(images)
        ce_loss = loss_func(outputs['fc'], targets)
        layer_list = ['layer4.0', 'layer4.1']
        var_loss = sum([torch.norm(outputs[layer]-self.ori_feature[layer], p=2)
                       for layer in layer_list])/len(layer_list)
        self.ce_loss = ce_loss.item()
        self.var_loss = var_loss.item()

        return ce_loss-self.beta*random.random()*var_loss

    def impl_loss_(self, model: nn.Module, images: torch.Tensor, targets: torch.Tensor, loss_func: torch.Tensor):
        outputs = model.forward(images)
        ce_loss = loss_func(outputs['fc'], targets)
        var_loss = self.make_var(outputs, targets)

        self.ce_loss = ce_loss.item()
        self.var_loss = var_loss.item()
        beta = 0.1
        return ce_loss+beta*var_loss

    def forward(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor, loss_func: nn.Module, target: None or int):

        model.eval()
        '''
        Creat a copy to prepare the generating of adversarial samples.
        '''
        ori_images = images.clone().to(self.device).detach()
        ori_images: torch.Tensor
        ori_images.requires_grad_(False)

        self.forzen(model)
        self.ori_feature = model.forward(ori_images)

        '''
        Select the target is the mode is targeted.
        '''
        targets = self.make_target(
            model, ori_images, labels, target).to(self.device)

        min_loss_item = -1 * np.inf

        for init_ in range(1, self.init_max+1):
            rand_ = 2*torch.randn(images.shape).to(self.device)-1
            rand_ *= self.epsilon
            images, _ = self.imaged_preprocess(
                ori_images+rand_, labels)

            images = self.multi_step_attack(
                model, images, ori_images, targets, loss_func)

            with torch.no_grad():
                loss = self.impl_loss(
                    model,
                    images,
                    targets,
                    loss_func
                )
                present_loss_item = loss.item()

            if present_loss_item >= min_loss_item:
                min_loss_item = present_loss_item
                adv_images = images

        self.deforzen(model)
        return adv_images
