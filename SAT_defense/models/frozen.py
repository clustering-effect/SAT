import torch.nn as nn
import torch
import torch.nn.functional as F

'''
resnet18
'''


class Frozen(nn.Module):
    def __init__(self, model, channels: list, num_classes=10):
        super().__init__()
        # Model type specifies number of layers for CIFAR-10 model
        self.frozen_model = model

        self.fcf256_15_1 = nn.Linear(channels[0]*2*2, 256)
        self.fcf256_16_1 = nn.Linear(channels[1]*2*2, 256)
        self.fcf256_17_1 = nn.Linear(channels[2]*2*2, 256)
        self.fcf256_18_1 = nn.Linear(channels[3]*2*2, 256)

        self.fcf256_15 = nn.Linear(256, num_classes)
        self.fcf256_16 = nn.Linear(256, num_classes)
        self.fcf256_17 = nn.Linear(256, num_classes)
        self.fcf256_18 = nn.Linear(256, num_classes)

    def load_frozen(self, filename, device=None):
        state = torch.load(filename, map_location=device)
        self.frozen_model.module.load_state_dict(state)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {
            k: v for k, v in state.items()
            if not k.startswith('frozen_model.')}

    def forward(self, x):
        outputs = self.frozen_model.forward(x)  # frozen model in eval mode
        f256_15, f256_16, f256_17, f256_18, f640, output_original = \
            outputs['layer4.0.conv1'], outputs['layer4.0.conv2'], \
            outputs['layer4.1.conv1'], outputs['layer4.1.conv2'], \
            outputs['avgpool'], outputs['fc']

        f256_15 = f256_15.view(f256_15.size(0), -1)
        f256_16 = f256_16.view(f256_16.size(0), -1)
        f256_17 = f256_17.view(f256_17.size(0), -1)
        f256_18 = f256_18.view(f256_18.size(0), -1)

        f256_15_1 = F.relu(self.fcf256_15_1(f256_15))
        f256_16_1 = F.relu(self.fcf256_16_1(f256_16))
        f256_17_1 = F.relu(self.fcf256_17_1(f256_17))
        f256_18_1 = F.relu(self.fcf256_18_1(f256_18))

        output_256_15 = self.fcf256_15(f256_15_1)
        output_256_16 = self.fcf256_16(f256_16_1)
        output_256_17 = self.fcf256_17(f256_17_1)
        output_256_18 = self.fcf256_18(f256_18_1)
        all_outputs = [output_256_15, output_256_16,
                       output_256_17, output_256_18, output_original]
        return all_outputs


'''
wideresnet
'''


# class Frozen(nn.Module):
#     def __init__(self, model, channels: list, num_classes=10):
#         super().__init__()
#         # Model type specifies number of layers for CIFAR-10 model
#         self.frozen_model = model

#         self.avgpool = nn.AvgPool2d(2)
#         # 8-->4
#         self.conv15_8_4 = nn.Conv2d(
#             channels[0], 64, kernel_size=2, stride=2, bias=False)
#         self.conv16_8_4 = nn.Conv2d(
#             channels[1], 64, kernel_size=2, stride=2, bias=False)
#         self.conv17_8_4 = nn.Conv2d(
#             channels[2], 64, kernel_size=2, stride=2, bias=False)
#         self.conv18_8_4 = nn.Conv2d(
#             channels[3], 64, kernel_size=2, stride=2, bias=False)
#         # 4-->2
#         self.conv15_4_2 = nn.Conv2d(
#             64, 64, kernel_size=2, stride=2, bias=False)
#         self.conv16_4_2 = nn.Conv2d(
#             64, 64, kernel_size=2, stride=2, bias=False)
#         self.conv17_4_2 = nn.Conv2d(
#             64, 64, kernel_size=2, stride=2, bias=False)
#         self.conv18_4_2 = nn.Conv2d(
#             64, 64, kernel_size=2, stride=2, bias=False)

#         self.fcf256_15_1 = nn.Linear(2048, 256)
#         self.fcf256_16_1 = nn.Linear(2048, 256)
#         self.fcf256_17_1 = nn.Linear(2048, 256)
#         self.fcf256_18_1 = nn.Linear(2048, 256)

#         self.fcf256_15 = nn.Linear(256, num_classes)
#         self.fcf256_16 = nn.Linear(256, num_classes)
#         self.fcf256_17 = nn.Linear(256, num_classes)
#         self.fcf256_18 = nn.Linear(256, num_classes)

#     def forward(self, x):
#         outputs = self.frozen_model.forward(x)  # frozen model in eval mode
#         f256_15, f256_16, f256_17, f256_18, f640, output_original = \
#             outputs['layer3.2.conv1'], outputs['layer3.2.conv2'], \
#             outputs['layer3.3.conv1'], outputs['layer3.3.conv2'], \
#             outputs['avgpool'], outputs['fc']

#         # 8-->4
#         fconv15_8_4 = F.relu(self.conv15_8_4(f256_15))
#         fconv16_8_4 = F.relu(self.conv16_8_4(f256_16))
#         fconv17_8_4 = F.relu(self.conv17_8_4(f256_17))
#         fconv18_8_4 = F.relu(self.conv18_8_4(f256_18))
#         # 4-->2
#         fconv15_2 = F.relu(self.conv15_4_2(fconv15_8_4))
#         fconv16_2 = F.relu(self.conv16_4_2(fconv16_8_4))
#         fconv17_2 = F.relu(self.conv17_4_2(fconv17_8_4))
#         fconv18_2 = F.relu(self.conv18_4_2(fconv18_8_4))

#         f256_15 = fconv15_2.view(fconv15_2.size(0), -1)
#         f256_16 = fconv16_2.view(fconv16_2.size(0), -1)
#         f256_17 = fconv17_2.view(fconv17_2.size(0), -1)
#         f256_18 = fconv18_2.view(fconv18_2.size(0), -1)

#         f256_15_1 = F.relu(self.fcf256_15_1(f256_15))
#         f256_16_1 = F.relu(self.fcf256_16_1(f256_16))
#         f256_17_1 = F.relu(self.fcf256_17_1(f256_17))
#         f256_18_1 = F.relu(self.fcf256_18_1(f256_18))

#         output_256_15 = self.fcf256_15(f256_15_1)
#         output_256_16 = self.fcf256_16(f256_16_1)
#         output_256_17 = self.fcf256_17(f256_17_1)
#         output_256_18 = self.fcf256_18(f256_18_1)
#         all_outputs = [output_256_15, output_256_16,
#                        output_256_17, output_256_18, output_original]
#         return all_outputs
