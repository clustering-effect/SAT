

if __name__ == '__main__':
    import torch
    from utils import *
    from main_config import model_config, dataset_config, auxiliary_config
    from torch.utils.data import DataLoader
    import copy
    import re
    from tqdm import tqdm
    from attacks import *
    import torch.nn as nn
    from cluster_compute import clus_compute

    '''
    Model preprocess.
    '''
    clus_compute()
    path = ''
    model = picker.model_picker(model_config)
    model.load_state_dict(torch.load(path)['state_dict'])
    model.to('cuda')
    model.eval()
    for _, param in model.named_parameters():
        param.requires_grad_(False)

    layers = dict(model.named_modules()).keys()
    num_classes = model_config['num_classes']
    p = model_config['p']
    extracted_features_ = torch.load(
        '')
    extractor = controller.FeatureExtractor(model, layers)

    '''
    Attack preprocess.
    '''
    dataset_packer = picker.dataset_picker(dataset_config)
    testset = dataset_packer['testset']
    testloader = DataLoader(testset, batch_size=500)

    attack = picker.attack_picker(auxiliary_config=auxiliary_config)
    loss = nn.CrossEntropyLoss()
    attack: pgd.PGD
    cluster_acc = {layer: 0 for layer in layers}
    classify_acc = 0

    for batch_idx, (img, tgt) in tqdm(enumerate(testloader), desc='Testing', colour='blue', total=len(testloader)):
        img, tgt = img.to('cuda'), tgt.to('cuda')
        batchsize = img.size(0)
        adv_img = attack['FGSM'](model, img, tgt, loss, None)

        outputs = extractor(adv_img)
        ori_outputs = extractor(img)

        pred = outputs['fc'].argmax(dim=1, keepdim=True)
        clean_num = pred.eq(tgt.view_as(pred)).sum().item()
        classify_acc += clean_num/len(testset)

        for layer, mid_output in outputs.items():
            distance = torch.norm(
                mid_output.cpu().view(batchsize, -1).unsqueeze(dim=1).repeat(1, num_classes, 1)
                -
                extracted_features_[layer].unsqueeze(dim=0).repeat(batchsize, 1, 1), dim=-1, p=p)
            pred = distance.argmin(dim=1, keepdim=True).cuda()
            cluster_acc[layer] += pred.eq(tgt.view_as(pred)
                                          ).sum().item()/len(testset)

    for layer, acc in cluster_acc.items():
        print('Clustering {} : {}%'.format(layer, 100*acc))

    print('Classify: {}%'.format(100*classify_acc))
