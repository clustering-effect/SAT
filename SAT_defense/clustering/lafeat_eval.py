import sys
sys.path.append('/home/jyl/SAT/')

if __name__ == '__main__':
    import torch
    from utils import *
    from main_config import model_config, dataset_config
    from torch.utils.data import DataLoader
    import copy
    import re
    from tqdm import tqdm
    from attacks.lafeat import LAFEAT
    import torch.nn as nn
    from models.frozen import Frozen
    from torch.optim import Adam

    '''
    Model preprocess.
    '''
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
    trainset = dataset_packer['trainset']
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    testset = dataset_packer['testset']
    testloader = DataLoader(testset, batch_size=500)

    '''
    Determine the size, find the key of the 4-lash Convolution layer.
    '''
    # random_input = torch.rand(1, 3, 32, 32).to('cuda')
    # random_output = extractor.forward(random_input)
    # for i, (k, v) in enumerate(random_output.items()):
    #     print(k, v.shape)
    linearlayer = Frozen(model=extractor, channels=[
                         512, 512, 512, 512]).to('cuda')

    '''
    Train the auxiliary linearlayer.
    '''
    loss = nn.CrossEntropyLoss()
    opt = Adam([p for p in linearlayer.parameters()
               if p.requires_grad], lr=0.001)

    for epoch_idx in range(100):
        loss_record = 0

        for batch_idx, (img, tgt) in enumerate(trainloader):
            opt.zero_grad()
            img, tgt = img.to('cuda'), tgt.to('cuda')
            outputs = linearlayer(img)
            all_loss = sum([loss(output, tgt) for output in outputs])
            all_loss.backward()
            opt.step()
            loss_record += all_loss.item()
        loss_record /= len(trainloader)

        print('Epoch: {}  all loss: {}'.format(
            epoch_idx+1, round(loss_record, 4)))
    torch.save(linearlayer.state_dict(),
               '')

    '''
    Load parameters.
    '''
    linearlayer.load_state_dict(torch.load(
        ''), strict=False)
    for name, param in linearlayer.named_parameters():
        param.requires_grad_(False)

    '''
    Generate lafeat samples.
    '''
    attack = LAFEAT(linearlayer, eps=8/255)
    cluster_acc = {layer: 0 for layer in layers}
    classify_acc = 0

    for batch_idx, (img, tgt) in enumerate(testloader):
        img, tgt = img.to('cuda'), tgt.to('cuda')
        batchsize = img.size(0)
        adv_img = attack.perturb(img, tgt)[1]
        outputs = extractor(adv_img)

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
