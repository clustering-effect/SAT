import torch
import torch.nn as nn
from utils.controller import *
from main_config import dataset_config, model_config



class _LOGGER():
    def __init__(self, attacks: list, train_size: int, test_size: int) -> None:
        self.clean_loss_logger = list()
        self.clean_acc_logger = list()

        self.adv_loss_logger = dict()
        self.adv_acc_logger = dict()

        self.attacks = attacks

        for attack in self.attacks:
            if attack == 'FAT' or attack == 'TRADES':
                continue
            self.adv_loss_logger[attack] = []
            self.adv_acc_logger[attack] = []

        self.train_size = train_size
        self.test_size = test_size
        self.makefile()

    def batch_logger_refresh(self):
        self.batch_clean_acc = list()
        self.batch_clean_loss = list()

        self.batch_adv_acc = dict()
        self.batch_adv_loss = dict()

        for attack in self.attacks:
            if attack == 'FAT' or attack == 'TRADES':
                continue
            self.batch_adv_acc[attack] = []
            self.batch_adv_loss[attack] = []

    def record_updater(self):
        self.acc_record_updater(self.batch_clean_acc, self.batch_adv_acc)
        self.loss_record_updater(self.batch_clean_loss, self.batch_adv_loss)

    def batch_acc_record_updater(self, clean_num: int, adv_num: dict):
        self.batch_clean_acc.append(clean_num)
        for attack in self.attacks:
            if attack == 'FAT' or attack == 'TRADES':
                continue
            self.batch_adv_acc[attack].append(adv_num[attack])

    def batch_loss_record_updater(self, clean_loss: float, adv_loss: dict, batchsize: int):
        self.batch_clean_loss.append(clean_loss*batchsize)
        for attack in self.attacks:
            if attack == 'FAT' or attack == 'TRADES':
                continue
            self.batch_adv_loss[attack].append(adv_loss[attack]*batchsize)

    def makefile(self, dataset: str = dataset_config['dataset'],
                 model: str = model_config['model'],
                 save_path: str = model_config['save_path']):

        self.save_path = save_path + '{}/{}/'.format(dataset, model)
        makedir(self.save_path)
        print('Save path created.')

    def acc_record_updater(self, batch_clean_acc: list, batch_adv_acc: dict):
        avg_clean_acc = sum(batch_clean_acc)/self.test_size
        avg_clean_acc *= 100
        self.clean_acc_logger.append(avg_clean_acc)

        attacks = self.adv_acc_logger.keys()
        for attack in attacks:
            if attack == 'FAT' or attack == 'TRADES':
                continue
            avg_adv_acc = sum(batch_adv_acc[attack])/self.test_size
            avg_adv_acc *= 100
            self.adv_acc_logger[attack].append(avg_adv_acc)

    def loss_record_updater(self, batch_clean_loss: list, batch_adv_loss: dict):
        avg_clean_loss = sum(batch_clean_loss)/self.test_size
        self.clean_loss_logger.append(avg_clean_loss)

        for attack in self.attacks:
            if attack == 'FAT' or attack == 'TRADES':
                continue
            avg_adv_loss = sum(batch_adv_loss[attack])/self.test_size
            self.adv_loss_logger[attack].append(avg_adv_loss)

    def model_record_update(self, model: nn.Module):
        self.state_dict = model.state_dict()

    def make_checkpoints(self, train_mode: str):
        ckpt = {
            'clean_loss': self.clean_loss_logger,
            'clean_acc': self.clean_acc_logger,
            'adv_loss': self.adv_loss_logger,
            'adv_acc': self.adv_acc_logger,
            'state_dict': self.state_dict
        }
        torch.save(ckpt, self.save_path+'{}.pth'.format(train_mode))
