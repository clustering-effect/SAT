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

EPS = 1E-20


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class AdvWeightPerturb(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, targets):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        loss = - F.cross_entropy(self.proxy(inputs_adv), targets)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)


sys.path.append('/home/jyl/SAT/')


class awpadvtrainer(_TRAIN):
    def __init__(self) -> None:
        super().__init__()
        self.mode = 'awpadvtrain'
        self.train_attack = 'PGD'

    def make_train_attacker(self, attacker: dict):

        train_attacker = attacker['PGD']
        train_attacker: PGD
        train_attacker.iter_max = 10
        return train_attacker

    def train(self):
        model, trainloader, evalloader, testloader, optimizer, scheduler = self.auxiliary_maker()
        proxy, _, _, _, proxy_opt, _ = self.auxiliary_maker()
        for param_group in proxy_opt.param_groups:
            param_group["lr"] = 0.01

        loss_func = nn.CrossEntropyLoss()
        attacker = self.make_attacker()
        train_attacker = self.make_train_attacker(attacker)
        awp_adversary = AdvWeightPerturb(model, proxy, proxy_opt, 0.001)

        logger = _LOGGER(self.attacks, self.train_size, self.test_size)
        c_acc = []
        for epoch_idx in range(1, self.epoch_num + 1):

            print('********************************** Begin the {}th epoch ***************************************'.format(epoch_idx))

            advtrainloss = []
            for _, (images, labels) in tqdm(enumerate(trainloader), desc='Training', colour='blue', total=len(trainloader)):

                optimizer.zero_grad()
                images, labels = images.to(self.device), labels.to(self.device)
                adv_images = train_attacker(
                    model, images, labels, loss_func, None)
                model.train()

                awp = awp_adversary.calc_awp(adv_images, labels)
                awp_adversary.perturb(awp)

                output = self._impl_output(model, adv_images)

                advloss = loss_func(output, labels)
                loss = advloss

                advtrainloss.append(advloss.item())
                self.model_updater(optimizer, loss)

            print('Train avg adv loss: {}'.format(
                round(sum(advtrainloss)/len(advtrainloss), 2)))

            self.test(model, loss_func, testloader, logger, attacker)
            self.epoch_results(logger)
            self.checkpoints_save(logger, 'awp_advtrain')
            scheduler.step()

            cluster_acc = clus_compute()
            c_acc.append(cluster_acc)
            torch.save(c_acc, 'c_acc4.pth')

    def __call__(self):
        self.train()


if __name__ == '__main__':
    trainer = awpadvtrainer()
    trainer()
