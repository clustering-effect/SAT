import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
import torch
from _train import _TRAIN
from _logger import _LOGGER
from tqdm import tqdm
from attacks.pgd import PGD
from clustering.cluster_compute import clus_compute
import sys
import copy
import numpy as np
from torch.autograd import Variable


def mart_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural -
                              epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)

    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(
        logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust

    return loss


class marttrainer(_TRAIN):
    def __init__(self) -> None:
        super().__init__()
        self.mode = 'marttrain'
        self.train_attack = 'PGD'

    def make_train_attacker(self, attacker: dict):

        train_attacker = attacker['PGD']
        train_attacker: PGD
        train_attacker.iter_max = 7
        return train_attacker

    def train(self):
        model, trainloader, evalloader, testloader, optimizer, scheduler = self.auxiliary_maker()
        loss_func = nn.CrossEntropyLoss()
        attacker = self.make_attacker()
        train_attacker = self.make_train_attacker(attacker)

        logger = _LOGGER(self.attacks, self.train_size, self.test_size)
        c_acc = []
        for epoch_idx in range(1, self.epoch_num + 1):

            print('********************************** Begin the {}th epoch ***************************************'.format(epoch_idx))

            advtrainloss = []
            for _, (images, labels) in tqdm(enumerate(trainloader), desc='Training', colour='blue', total=len(trainloader)):

                optimizer.zero_grad()
                images, labels = images.to(self.device), labels.to(self.device)
                model.train()

                advloss = mart_loss(model, images, labels, optimizer)
                loss = advloss

                advtrainloss.append(advloss.item())
                self.model_updater(optimizer, loss)

            print('Train avg adv loss: {}'.format(
                round(sum(advtrainloss)/len(advtrainloss), 2)))

            self.test(model, loss_func, testloader, logger, attacker)
            self.epoch_results(logger)
            self.checkpoints_save(logger, 'marttrain')
            scheduler.step()

            cluster_acc = clus_compute()
            c_acc.append(cluster_acc)
            torch.save(c_acc, 'c_acc2.pth')

    def __call__(self):
        self.train()


if __name__ == '__main__':
    trainer = marttrainer()
    trainer()
