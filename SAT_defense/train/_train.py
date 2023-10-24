

from torch.utils.data import DataLoader
from _logger import _LOGGER
from attacks._attack import attack
from tqdm import tqdm
from utils.picker import *
from utils.controller import *
from main_config import dataset_config, model_config, auxiliary_config
import torch.cuda
import torch


class _TRAIN():
    def __init__(
        self,
        dataset_config: dict = dataset_config,
        model_config: dict = model_config,
        auxiliary_config: dict = auxiliary_config
    ) -> None:

        self.dataset_config = dataset_config
        self.model_config = model_config
        self.auxiliary_config = auxiliary_config
        self.attacks = auxiliary_config['attacks']

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.info()

    def info(self):
        dataset = self.dataset_config['dataset']
        model = self.model_config['model']
        train_batch_size = self.dataset_config['train_batch_size']
        self.epoch_num = self.model_config['epoch_num']

        optimizer = self.auxiliary_config['opt']
        learning_rate = self.auxiliary_config['lr']
        momentum = self.auxiliary_config['momentum']
        weight_decay = self.auxiliary_config['weight_decay']

        scheduler = self.auxiliary_config['stl']
        milestones = self.auxiliary_config['milestones']
        gamma = self.auxiliary_config['gamma']

        print('********************************** Training Information ***************************************')
        print('Dataset: {}\t\tmodel: {}\t\ttrain_batch_size: {}\t\tepoch number: {}'.format(dataset, model,
                                                                                            train_batch_size, self.epoch_num))
        print('Optimizer: {}\t\t\tlearning rate: {}\t\tmomentum: {}\t\t\tweight_decay: {}'.format(optimizer, learning_rate,
                                                                                                  momentum, weight_decay))
        print('Scheduler: {}\t\tmilestones: [{}, {}]\t\tgamma: {}'.format(scheduler, milestones[0],
                                                                          milestones[1], gamma))
        print('Device: {}'.format(self.device))

    def auxiliary_maker(self, auxiliary_picker: callable = auxiliary_picker,
                        model_picker: callable = model_picker,
                        dataset_picker: callable = dataset_picker):

        model = model_picker(self.model_config)
        dataset_packer = dataset_picker(self.dataset_config)
        auxiliary = auxiliary_picker(self.auxiliary_config, model)

        self.train_size = dataset_packer['train_number']
        self.test_size = dataset_packer['test_number']

        trainloader = dataset_packer['trainloader']
        evalloader = dataset_packer['evalloader']
        testloader = dataset_packer['testloader']

        optimizer = auxiliary['opt']
        scheduler = auxiliary['stl']

        return model.to(self.device), trainloader, evalloader, testloader, optimizer, scheduler

    def train(self):
        raise NotImplementedError

    def test(self, model: nn.Module, loss_func: nn.Module,
             testloader: DataLoader, logger: _LOGGER, attacker: dict):
        model.eval()
        logger.batch_logger_refresh()

        for _, (images, labels) in tqdm(enumerate(testloader), desc='Testing', colour='green', total=len(testloader)):
            images, labels = images.to(self.device), labels.to(self.device)
            batchsize = images.shape[0]

            with torch.no_grad():
                output: torch.Tensor = self._impl_output(model, images)
                clean_loss = loss_func(output, labels).item()

                pred = output.argmax(dim=1, keepdim=True)
                clean_num = pred.eq(labels.view_as(pred)).sum().item()

            adv_num = {}
            adv_loss = {}
            for attack in self.attacks:
                if attack == 'FAT' or attack == 'TRADES':
                    continue
                attacker_: attack = attacker[attack]
                adv_images = attacker_(model, images, labels, loss_func, None)

                with torch.no_grad():
                    output: torch.Tensor = self._impl_output(model, adv_images)
                    adv_loss[attack] = loss_func(output, labels).item()

                    pred = output.argmax(dim=1, keepdim=True)
                    adv_num[attack] = pred.eq(
                        labels.view_as(pred)).sum().item()

            logger.batch_acc_record_updater(clean_num, adv_num)
            logger.batch_loss_record_updater(clean_loss, adv_loss, batchsize)

        logger.record_updater()
        logger.model_record_update(model)

    def epoch_results(self, logger: _LOGGER):
        clean_acc = logger.clean_acc_logger[-1]
        clean_loss = logger.clean_loss_logger[-1]
        print('Clean test acc: {}   Clean test loss: {}'.format(
            round(clean_acc, 2), round(clean_loss, 2)
        ))

        for attack in self.attacks:
            if attack == 'FAT' or attack == 'TRADES':
                continue
            adv_acc = logger.adv_acc_logger[attack][-1]
            adv_loss = logger.adv_loss_logger[attack][-1]
            print('Attack: {}   Adv test acc: {}    Adv test loss: {}'.format(
                attack, round(adv_acc, 2), round(adv_loss, 2)
            ))

    def checkpoints_save(self, logger: _LOGGER, mode: str):
        logger.make_checkpoints(mode)

    def model_updater(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _impl_output(self, model: nn.Module, images: torch.Tensor):
        output = model(images)

        if isinstance(output, torch.Tensor):
            pass

        elif isinstance(output, list):
            output = output[0]

        return output

    def make_attacker(self):
        attacker = attack_picker(self.auxiliary_config)
        return attacker

    def trained_model_eval(self, mode: str):
        model, _, evalloader, testloader, _, _ = self.auxiliary_maker()
        loss_func = nn.CrossEntropyLoss()
        attacker = self.make_attacker()
        logger = _LOGGER(self.attacks, self.train_size, self.test_size)
        save_path = logger.save_path
        model.load_state_dict(torch.load(
            save_path+'{}.pth'.format(mode))['state_dict'])
        self.test(model, loss_func, testloader, logger, attacker)
        self.epoch_results(logger)

        self.test(model, loss_func, evalloader, logger, attacker)
        self.epoch_results(logger)

  