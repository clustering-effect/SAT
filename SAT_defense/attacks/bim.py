import torch
import torch.nn as nn
from fgsm import FGSM

'''
epsilon: The size of the perturbation.
alpha: The size of each step.
is_nontarget: Indicate the mode of attack, if True, nontargeted, else targeted.

'''
class BIM(FGSM):
    def __init__(self, epsilon: float = 8 / 255, 
                        alpha: float = 1 / 255, 
                        iter_max: int = 50,
                        device: float = 'cuda', 
                        is_nontarget: bool = True) -> None:
        super().__init__(epsilon, device, is_nontarget)

        self.alpha = alpha
        self.iter_max = iter_max


    def forward(self, model:nn.Module, images:torch.Tensor, labels:torch.Tensor, loss_func:nn.Module, target:None or int):
        '''
        Creat a copy to prepare the generating of adversarial samples.
        '''
        ori_images = images.clone().to(self.device).detach()
        images, labels = self.imaged_preprocess(images, labels)

        self.forzen(model)

        '''
        Select the target is the mode is targeted.
        '''
        targets = self.make_target(model, images, labels, target)
        images = self.multi_step_attack(model, images, ori_images, targets, loss_func)
        
        self.deforzen(model)
        adv_images = images
        return adv_images.requires_grad_(False)


    def multi_step_attack(self, model:nn.Module, images:torch.Tensor, ori_images:torch.Tensor,\
                                                     targets:torch.Tensor, loss_func:nn.Module):
        for _ in range(1, self.iter_max+1):
            images = self.one_step_attack(model, images, ori_images,\
                                        targets, loss_func, self.alpha)
        return images

        