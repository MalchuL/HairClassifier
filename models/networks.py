import torch
import torch.nn as nn
import torchvision.models as models

import registry.registries as registry
from efficientnet_pytorch import EfficientNet

registry.Model(EfficientNet.from_pretrained, name='EfficientNet')


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class ShuffleNetV2(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        assert in_channels == 3
        self.model = models.shufflenet_v2_x1_0(True)

        set_requires_grad(self.model, True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.train()
    def forward(self, x):
        x = self.model(x)

        return x


class MobileNetV3(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        assert in_channels == 3
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v3_small', pretrained=True)

        set_requires_grad(self.model, True)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)
        self.model.train()

    def forward(self, x):
        x = self.model(x)

        return x

registry.Model(ShuffleNetV2)
registry.Model(MobileNetV3)