
from _train import _TRAIN
from _logger import _LOGGER
import torch.nn as nn
from tqdm import tqdm
from attacks.pgd import PGD
from clustering.cluster_compute import clus_compute
import torch


class advtrainer(_TRAIN):
    def __init__(self) -> None:
        super().__init__()
        self.mode = 'advtrain'
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
                adv_images = train_attacker(
                    model, images, labels, loss_func, None)
                model.train()

                output = self._impl_output(model, adv_images)

                advloss = loss_func(output, labels)
                loss = advloss

                advtrainloss.append(advloss.item())
                self.model_updater(optimizer, loss)

            print('Train avg adv loss: {}'.format(
                round(sum(advtrainloss)/len(advtrainloss), 2)))

            self.test(model, loss_func, testloader, logger, attacker)
            self.epoch_results(logger)
            self.checkpoints_save(logger, 'advtrain')
            scheduler.step()

            cluster_acc = clus_compute()
            c_acc.append(cluster_acc)
            torch.save(c_acc, 'c_acc0.pth')

    def __call__(self):
        self.train()


if __name__ == '__main__':
    trainer = advtrainer()
    trainer()
