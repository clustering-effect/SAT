import sys
sys.path.append('/home/jyl/SAT/')

if __name__ == '__main__':
    import torch
    from utils import *
    from main_config import model_config, dataset_config, auxiliary_config
    from torch.utils.data import DataLoader
    import copy
    import re
    from tqdm import tqdm
    from attacks.lafeat import LAFEAT
    import torch.nn as nn
    from models.frozen import Frozen
    import matplotlib.pyplot as plt
    '''
    Model preprocess.
    '''
    path = '/home/jyl/SAT/checkpoints/CIFAR10/resnet18/advtrain.pth'
    model = picker.model_picker(model_config)
    model.load_state_dict(torch.load(path)['state_dict'])
    model.to('cuda')
    model.eval()
    for _, param in model.named_parameters():
        param.requires_grad_(False)

    layers = dict(model.named_modules()).keys()
    num_classes = model_config['num_classes']
    p = model_config['p']

    '''
    Attack preprocess.
    '''
    dataset_packer = picker.dataset_picker(dataset_config)
    trainset = dataset_packer['trainset']
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    testset = dataset_packer['testset']
    testloader = DataLoader(testset, batch_size=500)

    extractor = controller.FeatureExtractor(model, layers)
    linearlayer = Frozen(model=extractor, channels=[
                         512, 512, 512, 512]).to('cuda')
    linearlayer.load_state_dict(torch.load(
        '/home/jyl/SAT/checkpoints/CIFAR10/resnet18/advtrain_aug.pth'), strict=False)
    linearlayer.eval()
    attack = picker.attack_picker(auxiliary_config=auxiliary_config)

    ila_attack = LAFEAT(linearlayer, eps=8/255)
    loss = nn.CrossEntropyLoss()

    differs = []

    for batch_idx, (img, tgt) in tqdm(enumerate(testloader), desc='Testing', colour='blue', total=len(testloader)):
        img, tgt = img.to('cuda'), tgt.to('cuda')
        batchsize = img.size(0)
        adv_img = attack['PGD'](model, img, tgt, loss,
                                None).view(batchsize, -1)

        ila_img = attack['BIM'](model, img, tgt, loss,
                                None).view(batchsize, -1)

        differs += torch.norm(adv_img - ila_img, p=p, dim=1).view(-1).tolist()

    import numpy as np
    upper = -np.inf
    lower = np.inf

    for differ in differs:
        if differ >= upper:
            upper = differ
        if differ <= lower:
            lower = differ

    rate = {i: 0 for i in range(0, 101)}
    for differ in differs:
        posi = round(100*(differ-lower)/(upper-lower))
        rate[posi] += round(100/len(differs), 3)
    print(upper)
    print(lower)
    print(rate)

    x = [j for j in range(0, 101)]
    y = list(rate.values())

    plt.bar(x, y)
    plt.savefig('bar.png')
