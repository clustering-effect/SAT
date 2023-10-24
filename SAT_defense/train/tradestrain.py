from _train import _TRAIN
from _logger import _LOGGER
import torch.nn as nn
from tqdm import tqdm
from utils import *
from attacks.trades import TRADES
import sys
import torch
import numpy as np
from clustering.cluster_compute import clus_compute


class tradestrainer(_TRAIN):
    def __init__(self) -> None:
        super().__init__()
        self.mode = 'tradestrain'
        self.train_attack = ''

    def make_train_attacker(self, attacker: dict):

        train_attacker = attacker['TRADES']
        train_attacker: TRADES
        train_attacker.iter_max = 10
        return train_attacker

    def make_attacker(self):
        attacker = picker.attack_picker(self.auxiliary_config)
        return attacker

    def train(self):
        model, trainloader, evalloader, testloader, optimizer, scheduler = self.auxiliary_maker()
        kl_loss_func = nn.KLDivLoss(size_average=False)
        loss_func = nn.CrossEntropyLoss()
        attacker = self.make_attacker()
        train_attacker = self.make_train_attacker(attacker)
        train_attacker.fat = False

        logger = _LOGGER(self.attacks, self.train_size, self.test_size)

        layers = dict(model.named_modules()).keys()
        extractor = controller.FeatureExtractor(model, layers)
        c_acc = []
        for epoch_idx in range(1, self.epoch_num + 1):

            print('********************************** Begin the {}th epoch ***************************************'.format(epoch_idx))

            advtrainloss = []
            for _, (images, labels) in tqdm(enumerate(trainloader), desc='Training', colour='blue', total=len(trainloader)):

                optimizer.zero_grad()
                images, labels = images.to(self.device), labels.to(self.device)
                ori_images = images.clone()
                batchsize = images.shape[0]
                adv_images = train_attacker(
                    extractor, images, labels, kl_loss_func, None)
                extractor.model.train()

                advloss = train_attacker.impl_loss(
                    extractor, ori_images, adv_images, labels, kl_loss_func)/batchsize
                loss = advloss + \
                    loss_func(extractor.forward(adv_images)['fc'], labels)

                advtrainloss.append(loss.item())
                self.model_updater(optimizer, loss)

            print('Train avg adv loss: {}'.format(
                round(sum(advtrainloss)/len(advtrainloss), 2)))

            self.test(extractor.model, loss_func, testloader, logger, attacker)
            self.epoch_results(logger)
            self.checkpoints_save(logger, 'tradestrain')
            scheduler.step()
            cluster_acc = clus_compute()
            c_acc.append(cluster_acc)
            torch.save(c_acc, 'c_acc1.pth')

    def __call__(self):
        self.train()


if __name__ == '__main__':
    trainer = tradestrainer()
    trainer()
