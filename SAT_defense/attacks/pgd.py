import numpy as np
from bim import BIM
import torch.nn as nn
import torch
import sys
from torch.autograd import grad


class PGD(BIM):

    def __init__(self, epsilon: float = 8 / 255,
                 alpha: float = 2 / 255,
                 init_max: int = 1,
                 iter_max: int = 100,
                 device: float = 'cuda',
                 is_nontarget: bool = True) -> None:
        super().__init__(epsilon, alpha, iter_max, device, is_nontarget)
        self.alpha = alpha
        self.init_max = init_max
        self.iter_max = iter_max

    def forward(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor, loss_func: nn.Module, target: None or int = None):

        model.eval()
        '''
        Creat a copy to prepare the generating of adversarial samples.
        '''
        ori_images = images.clone().to(self.device).detach()
        ori_images: torch.Tensor
        ori_images.requires_grad_(False)

        self.forzen(model)

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
        return adv_images.requires_grad_(False)

    def forward_one_by_one(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor, loss_func: nn.Module, target: None or int):

        model.eval()
        '''
        Creat a copy to prepare the generating of adversarial samples.
        '''
        ori_images = images.clone().to(self.device).detach()
        ori_images: torch.Tensor
        ori_images.requires_grad_(False)

        self.forzen(model)

        '''
        Select the target is the mode is targeted.
        '''
        targets = self.make_target(
            model, ori_images, labels, target).to(self.device)

        min_loss_item = -1 * np.inf

        for init_ in range(1, self.init_max+1):
            rand_ = 2*torch.randn(images.shape).to(self.device)-1
            rand_ *= 0
            images, _ = self.imaged_preprocess(
                ori_images+rand_, labels)

            while True:
                images = self.one_step_attack(model, images, ori_images,
                                              targets, loss_func, self.alpha)
                yield images

        self.deforzen(model)
