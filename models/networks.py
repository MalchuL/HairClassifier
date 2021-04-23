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


def on_load_checkpoint(new_model, old_model) -> None:
    state_dict = old_model.state_dict()
    model_state_dict = new_model.state_dict()
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print(f"Skip loading parameter: {k}, "
                            f"required shape: {model_state_dict[k].shape}, "
                            f"loaded shape: {state_dict[k].shape}")
                state_dict[k] = model_state_dict[k]

        else:
            print(f"Dropping parameter {k}")

    return state_dict

class ShuffleNetV2(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        assert in_channels == 3
        pretrained_model = models.shufflenet_v2_x0_5(True)

        self.model = models.ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 256], num_classes=num_classes)
        new_state_dict = on_load_checkpoint(self.model, pretrained_model)
        self.model.load_state_dict(new_state_dict)

        set_requires_grad(self.model, True)
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