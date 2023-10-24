import torch.autograd.functional as func
import torch
import numpy
dataset_config = {
    'dataset': 'CIFAR10',
    'dataset_save_path': '/home/data/',
    'size': [32, 32],
    'train_batch_size': 128,
    'test_batch_size': 500
}

model_config = {
    'model': 'resnet18',
    'num_classes': 10,
    'p': numpy.inf,
    'save_path': '',
    'epoch_num': 200
}

auxiliary_config = {
    'opt': 'SGD',
    'stl': 'MultiStepLR',

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,

    'milestones': [100, 150],
    'gamma': 0.1,

    'attacks': ['PGD', 'FAT']
}
