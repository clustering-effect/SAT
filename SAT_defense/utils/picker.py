from torchvision import datasets, transforms
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from torch.utils.data import DataLoader
from models import resnet, wideresnet
from attacks import fgsm, bim, pgd, fat, trades
import sys
sys.path.append('/home/jyl/SAT/')


def dataset_picker(dataset_config: dict):

    dataset_name = dataset_config['dataset']
    dataset_save_path = dataset_config['dataset_save_path']
    dataset_size = dataset_config['size']
    train_batch_size = dataset_config['train_batch_size']
    test_batch_size = dataset_config['test_batch_size']

    if dataset_name == 'MNIST':

        train_data_transform = transforms.Compose(
            [
                transforms.Resize(dataset_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )
        trainset = datasets.MNIST(
            dataset_save_path,
            train=True,
            transform=train_data_transform
        )

        test_data_transform = transforms.Compose(
            [
                transforms.Resize(dataset_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )
        testset = datasets.MNIST(
            dataset_save_path,
            train=False,
            transform=test_data_transform
        )

    elif dataset_name == 'FashionMNIST':

        train_data_transform = transforms.Compose(
            [
                transforms.Resize(dataset_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )
        trainset = datasets.FashionMNIST(
            dataset_save_path,
            train=True,
            transform=train_data_transform
        )

        test_data_transform = transforms.Compose(
            [
                transforms.Resize(dataset_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )
        testset = datasets.FashionMNIST(
            dataset_save_path,
            train=False,
            transform=test_data_transform
        )

    elif dataset_name == 'CIFAR10':

        train_data_transform = transforms.Compose(
            [
                transforms.Resize(dataset_size),
                transforms.RandomCrop(dataset_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        trainset = datasets.CIFAR10(
            dataset_save_path,
            train=True,
            transform=train_data_transform
        )

        test_data_transform = transforms.Compose(
            [
                transforms.Resize(dataset_size),
                transforms.ToTensor(),
            ]
        )

        testset = datasets.CIFAR10(
            dataset_save_path,
            train=False,
            transform=test_data_transform
        )

    elif dataset_name == 'SVHN':

        train_data_transform = transforms.Compose(
            [
                transforms.Resize(dataset_size),
                transforms.RandomCrop(dataset_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        trainset = datasets.SVHN(
            '/home/data/SVHN',
            split='train',
            transform=train_data_transform
        )

        test_data_transform = transforms.Compose(
            [
                transforms.Resize(dataset_size),
                transforms.ToTensor(),
            ]
        )

        testset = datasets.SVHN(
            '/home/data/SVHN',
            split='test',
            transform=test_data_transform
        )

    elif dataset_name == 'CIFAR100':

        train_data_transform = transforms.Compose(
            [
                transforms.Resize(dataset_size),
                transforms.RandomCrop(dataset_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ]
        )

        trainset = datasets.CIFAR100(
            dataset_save_path,
            train=True,
            transform=train_data_transform
        )

        test_data_transform = transforms.Compose(
            [
                transforms.Resize(dataset_size),
                transforms.ToTensor(),
            ]
        )

        testset = datasets.CIFAR100(
            dataset_save_path,
            train=False,
            transform=test_data_transform
        )

    elif dataset_name == 'ImageNet':

        train_data_transform = transforms.Compose(
            [
                transforms.Resize(dataset_size),
                transforms.RandomCrop(dataset_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ]
        )

        trainset = datasets.ImageFolder(
            root=dataset_save_path,
            transform=train_data_transform
        )

        test_data_transform = transforms.Compose(
            [
                transforms.Resize(dataset_size),
                transforms.ToTensor(),
            ]
        )

        testset = datasets.ImageFolder(
            root=dataset_save_path,
            transform=test_data_transform
        )

    return {
        'trainset': trainset,
        'trainloader': DataLoader(trainset, train_batch_size, shuffle=True),
        'evalloader': DataLoader(trainset, test_batch_size),
        'train_number': len(trainset),
        'testset': testset,
        'testloader': DataLoader(testset, test_batch_size),
        'test_number': len(testset)
    }


def auxiliary_picker(auxiliary_config: dict, model: nn.Module):
    opt_name = auxiliary_config['opt']
    stl_name = auxiliary_config['stl']
    lr = auxiliary_config['lr']
    momentum = auxiliary_config['momentum']
    weight_decay = auxiliary_config['weight_decay']
    milestones = auxiliary_config['milestones']
    gamma = auxiliary_config['gamma']

    if opt_name == 'SGD' and stl_name == 'MultiStepLR':
        opt = SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        stl = MultiStepLR(
            opt,
            milestones=milestones,
            gamma=gamma
        )
    else:
        raise NotImplementedError('You haven\'t realize this auxiliary!')

    return {
        'opt': opt,
        'stl': stl
    }


def model_picker(model_config: dict):
    num_classes = model_config['num_classes']
    model_name = model_config['model']

    if model_name == 'resnet18':
        model = resnet.ResNet18(num_classes)

    elif model_name == 'resnet34':
        model = resnet.ResNet34(num_classes)

    elif model_name == 'resnet50':
        model = resnet.ResNet50(num_classes)

    elif model_name == 'resnet101':
        model = resnet.ResNet101(num_classes)

    elif model_name == 'resnet152':
        model = resnet.ResNet152(num_classes)

    elif model_name == 'wideresnet':
        model = wideresnet.wideResnet28_10(num_classes)

    return model


def attack_picker(auxiliary_config: dict):

    fgsm_config = {
        'epsilon': 8 / 255,
        'device': 'cuda',
        'is_nontarget': True
    }

    bim_config = {
        'epsilon': 8 / 255,
        'alpha': 2 / 255,
        'iter_max': 100,
        'device': 'cuda',
        'is_nontarget': True
    }

    pgd_config = {
        'epsilon': 8 / 255,
        'alpha': 2 / 255,
        'init_max': 1,
        'iter_max': 100,
        'device': 'cuda',
        'is_nontarget': True
    }

    fat_config = {
        'epsilon': 8 / 255,
        'alpha': 2 / 255,
        'init_max': 1,
        'iter_max': 100,
        'device': 'cuda',
        'is_nontarget': True
    }

    trades_config = {
        'epsilon': 8/255,
        'alpha': 2/255,
        'init_max': 1,
        'iter_max': 7,
        'device': 'cuda',
        'is_nontarget': True
    }
    attacker = {}
    attacks = auxiliary_config['attacks']
    for attack in attacks:
        if attack == 'FGSM':
            attacker[attack] = fgsm.FGSM(
                fgsm_config['epsilon'],
                fgsm_config['device'],
                fgsm_config['is_nontarget']
            )

        elif attack == 'BIM':
            attacker[attack] = bim.BIM(
                bim_config['epsilon'],
                bim_config['alpha'],
                bim_config['iter_max'],
                bim_config['device'],
                bim_config['is_nontarget']
            )

        elif attack == 'PGD':
            attacker[attack] = pgd.PGD(
                pgd_config['epsilon'],
                pgd_config['alpha'],
                pgd_config['init_max'],
                pgd_config['iter_max'],
                pgd_config['device'],
                pgd_config['is_nontarget']
            )

        elif attack == 'FAT':
            attacker[attack] = fat.FAT(
                fat_config['epsilon'],
                fat_config['alpha'],
                fat_config['init_max'],
                fat_config['iter_max'],
                fat_config['device'],
                fat_config['is_nontarget']
            )
        elif attack == 'TRADES':
            attacker[attack] = trades.TRADES(
                trades_config['epsilon'],
                trades_config['alpha'],
                trades_config['init_max'],
                trades_config['iter_max'],
                trades_config['device'],
                trades_config['is_nontarget']
            )

    return attacker
