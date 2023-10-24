import torch
import torch.nn as nn

class attack():
    def __init__(self, epsilon:float = 8/255, device:float = 'cuda', is_nontarget:bool = True) -> None:
        self.epsilon = epsilon
        self.device = device
        if is_nontarget:
            self.mode = 1
        else:
            self.mode = -1

    def forward(self,):
        raise NotImplementedError

    '''
    Method to generate new differential images and labels.
    '''
    def imaged_preprocess(self, images:torch.Tensor, labels:torch.Tensor):
        images = images.clone().to(self.device).clamp_(0., 1.).detach()
        labels = labels.clone().to(self.device).detach()
        images.requires_grad = True
        return images, labels

    '''
    Generate specific target labels.
    '''
    def obtain_target(self, target:int, label:torch.Tensor):
        target = torch.full(label.shape, target, device=label.device)
        return target

    '''
    Generate llt target labels.
    '''
    def least_like_target(self, model:nn.Module, images:torch.Tensor):
        with torch.no_grad():
            output = model(images)
            target = output.argmin(dim=1)
        return target

    '''
    forzen and deforzen to accelerate adversarial attacks.
    '''
    def forzen(self, model:nn.Module):
        for _, params in model.named_parameters():
            params.requires_grad_(False)
    
    def deforzen(self, model:nn.Module):
        for _, params in model.named_parameters():
            params.requires_grad_(True)
    
    '''
    Compute targets
    '''
    def make_target(self, model:nn.Module, images:torch.Tensor, labels:torch.Tensor, target:None or int):
        if self.mode == -1:
            if target == None:
                targets = self.least_like_target(model, images)
            elif isinstance(target, int):
                targets = self.obtain_target(target, labels)
            else:
                raise ValueError('Wrong data type for target!')

        elif self.mode == 1:
            targets = labels

        return targets
    
    '''
    Compute loss
    '''
    def impl_loss(self, model:nn.Module, images:torch.Tensor, targets:torch.Tensor, loss_func:torch.Tensor):
        output = model(images)

        if isinstance(output, torch.Tensor):
            loss = self.mode*loss_func(output, targets)

        elif isinstance(output, list):
            loss = self.mode*loss_func(output[0], targets)
        
        return loss
    
    def images_clamp(self, images:torch.Tensor, ori_images:torch.Tensor):
        images.clamp_(ori_images-self.epsilon, ori_images+self.epsilon)
        images.clamp_(0., 1.)
        images.detach_().requires_grad_(True)
        return images
    
    
    def __call__(self, model:nn.Module, images:torch.Tensor, labels:torch.Tensor, loss_func:nn.Module, target:None or int) -> torch.Tensor:
        return self.forward(model, images, labels, loss_func, target)
