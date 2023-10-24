import pandas as pd
from tqdm import tqdm
import re
import copy
from torch.utils.data import DataLoader
from main_config import model_config, dataset_config
from utils import *
import torch
import sys
sys.path.append('/home/jyl/SAT/')


'''
Model preprocess.
'''


def clus_compute():
    path = '/home/jyl/SAT/checkpoints/CIFAR10/resnet18/sattrain.pth'
    model = picker.model_picker(model_config)
    model.load_state_dict(torch.load(path)['state_dict'])
    model.to('cuda')
    model.eval()
    for _, param in model.named_parameters():
        param.requires_grad_(False)

    layers = dict(model.named_modules()).keys()

    extractor = controller.FeatureExtractor(model, layers)

    dataset_packer = picker.dataset_picker(dataset_config)
    trainset = dataset_packer['trainset']
    testset = dataset_packer['testset']
    num_classes = model_config['num_classes']

    labels = trainset.targets
    labels_ = ''.join(str(label) for label in labels)

    '''
    Compute clustering centroid.
    '''
    extracted_features = {}

    for i in range(num_classes):
        trainset_ = copy.deepcopy(trainset)
        idx = [idx_.start()
               for idx_ in re.finditer('{}'.format(i), labels_)]
        data = trainset.data
        trainset_.data = [data[j] for j in idx]
        trainset_.targets = [i]*len(trainset_.data)
        trainloader = DataLoader(trainset_, batch_size=100)
        features = {layer: [] for layer in layers}

        for batch_idx, (img, tgt) in enumerate(trainloader):
            img, tgt = img.to('cuda'), tgt.to('cuda')
            outputs = extractor.forward(img)
            output = outputs['fc']
            mask = output.argmax(dim=1, keepdim=True).view_as(tgt) == tgt

            for layer, mid_output in outputs.items():
                features[layer].append(mid_output[mask].sum(
                    dim=0).view(1, -1).cpu().detach())

        for layer in layers:
            features[layer] = sum(features[layer])/len(idx)

        extracted_features[i] = features

    '''
    Transform the save format.
    '''
    extracted_features_ = {layer: 0 for layer in layers}
    for layer in layers:
        extracted_features_[layer] = torch.cat(
            [extracted_features[i][layer] for i in range(num_classes)], dim=0)

    torch.save(extracted_features_,
               '/home/jyl/SAT/checkpoints/CIFAR10/resnet18/sattrain_cluster.pth')

    p = model_config['p']
    testloader = DataLoader(testset, batch_size=100)
    cluster_acc = {layer: 0 for layer in layers}

    for batch_idx, (img, tgt) in tqdm(enumerate(testloader), desc='Testing', colour='blue', total=len(testloader)):
        img, tgt = img.to('cuda'), tgt.to('cuda')
        batchsize = img.shape[0]
        outputs = extractor.forward(img)

        for layer, mid_output in outputs.items():
            distance = torch.norm(
                mid_output.cpu().view(batchsize, -1).unsqueeze(dim=1).repeat(1, num_classes, 1)
                -
                extracted_features_[layer].unsqueeze(dim=0).repeat(batchsize, 1, 1), dim=-1, p=p)
            pred = distance.argmin(dim=1, keepdim=True).cuda()
            cluster_acc[layer] += pred.eq(tgt.view_as(pred)
                                          ).sum().item()/len(testset)

    for layer, acc in cluster_acc.items():
        print('{} : {}%'.format(layer, 100*acc))
    return cluster_acc
