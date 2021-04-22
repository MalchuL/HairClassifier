import registry.registries as registry
from efficientnet_pytorch import EfficientNet

registry.Model(EfficientNet.from_pretrained, name='EfficientNet')