from _attack import attack
from torch.autograd import grad
import torch.nn as nn
import torch


'''
epsilon: The size of the perturbation.
is_nontarget: Indicate the mode of attack, if True, nontargeted, else targeted.
'''


class FGSM(attack):
    def __init__(self, epsilon: float = 8 / 255,
                 device: float = 'cuda',
                 is_nontarget: bool = True) -> None:
        super().__init__(epsilon, device, is_nontarget)

    def forward(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor, loss_func: nn.Module, target: None or int):
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
        adv_images = self.one_step_attack(
            model, images, ori_images, targets, loss_func, self.epsilon)
        self.deforzen(model)

        return adv_images.requires_grad_(False)

    def one_step_attack(self, model: nn.Module, images: torch.Tensor, ori_images: torch.Tensor,
                        targets: torch.Tensor, loss_func: nn.Module, step_size: float):
        '''
        Compute the loss function.
        '''
        model.eval()
        loss = self.impl_loss(model, images, targets, loss_func)

        '''
        Compute the gradient of x.
        '''
        perb = grad(loss, images, retain_graph=False,
                    create_graph=False)[0].sign()

        adv_images = images + perb*step_size
        adv_images = self.images_clamp(adv_images, ori_images)
        return adv_images
