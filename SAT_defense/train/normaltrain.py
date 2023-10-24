from _train import _TRAIN
from _logger import _LOGGER
import torch.nn as nn
from tqdm import tqdm
from clustering.cluster_compute import clus_compute
import torch


class normaltrainer(_TRAIN):
    def __init__(self) -> None:
        super().__init__()
        self.mode = 'normaltrain'

    def train(self):
        model, trainloader, evalloader, testloader, optimizer, scheduler = self.auxiliary_maker()
        loss_func = nn.CrossEntropyLoss()
        attacker = self.make_attacker()
        logger = _LOGGER(self.attacks, self.train_size, self.test_size)
        c_acc = []
        for epoch_idx in range(1, self.epoch_num + 1):

            print('********************************** Begin the {}th epoch ***************************************'.format(epoch_idx))

            trainloss = []
            for _, (images, labels) in tqdm(enumerate(trainloader), desc='Training', colour='blue', total=len(trainloader)):
                model.train()
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = self._impl_output(model, images)
                loss = loss_func(output, labels)
                trainloss.append(loss.item())

                self.model_updater(optimizer, loss)

            self.test(model, loss_func, testloader, logger, attacker)
            self.epoch_results(logger)
            self.checkpoints_save(logger, 'normaltrain')
            scheduler.step()
            cluster_acc = clus_compute()
            c_acc.append(cluster_acc)
            torch.save(c_acc, 'c_acc3.pth')

    def __call__(self):
        self.train()


if __name__ == '__main__':
    trainer = normaltrainer()
    trainer()
