from pgd import PGD
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import grad
import numpy as np


class TRADES(PGD):
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
        if is_nontarget:
            self.mode = 1
        else:
            self.mode = -1
        self.fat = False

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
        layer_list = ['layer4.1.conv1']

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

    def impl_loss(self, model: nn.Module, ori_images: torch.Tensor, images: torch.Tensor,
                  targets: torch.Tensor, loss_func: torch.nn.Module):
        outputs = model.forward(images)
        output = outputs['fc']
        ori_outputs = model.forward(ori_images)
        ori_output = ori_outputs['fc']
        loss = loss_func(F.log_softmax(output, dim=1),
                         F.softmax(ori_output, dim=1))
        if self.fat:
            beta = 0.5
            var_loss = self.make_var(outputs, targets)
            return loss + beta*var_loss

        return loss

    def one_step_attack(self, model: nn.Module, images: torch.Tensor, ori_images: torch.Tensor, targets: torch.Tensor, loss_func: nn.Module, step_size: float):
        '''
        Compute the loss function.
        '''
        model.eval()
        loss = self.impl_loss(model, ori_images, images, targets, loss_func)

        '''
        Compute the gradient of x.
        '''
        perb = grad(loss, images, retain_graph=False,
                    create_graph=False)[0].sign()

        adv_images = images + perb*step_size
        adv_images = self.images_clamp(adv_images, ori_images)
        return adv_images

    def multi_step_attack(self, model: nn.Module, images: torch.Tensor, ori_images: torch.Tensor,
                          targets: torch.Tensor, loss_func: nn.Module):
        for _ in range(1, self.iter_max+1):
            images = self.one_step_attack(model, images, ori_images,
                                          targets, loss_func, self.alpha)
        return images

    def forward(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor, loss_func: nn.Module, target: None or int):

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
            rand_ = 2*torch.rand(images.shape).to(self.device)-1
            rand_ *= 0.001
            images, _ = self.imaged_preprocess(
                ori_images+rand_, labels)

            images = self.multi_step_attack(
                model, images, ori_images, targets, loss_func)

            with torch.no_grad():
                loss = self.impl_loss(
                    model,
                    ori_images,
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
