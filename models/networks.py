import torch
import torch.nn as nn

import registry.registries as registry
from efficientnet_pytorch import EfficientNet

registry.Model(EfficientNet.from_pretrained, name='EfficientNet')


class ShuffleNetV2(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        assert in_channels == 3
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'shufflenet_v2_x0_5', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.train()
    def forward(self, x):
        x = self.model(x)

        return x

registry.Model(ShuffleNetV2)