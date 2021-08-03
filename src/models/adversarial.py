import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod
import segmentation_models_pytorch as smp


class AdversarialMaskTransform(nn.Module):
    def __init__(self, num_samples, output_size):
        super(AdversarialMaskTransform, self).__init__()
        self.num_samples = num_samples
        self.output_size = output_size
        self.embedding = nn.Embedding(num_samples, prod(output_size))

    def forward(self, x, idx=None):
        if idx is None:
            mask = self.embedding.weight[:x.size(0)]
        else:
            mask = self.embedding(idx)
        mask = mask.sigmoid()
        return x * mask.view_as(x)


class AdversarialUnetTransform(smp.Unet):
    def __init__(self, encoder_name="resnet18", in_channels=3, classes=3, output_init_bias=1.,
                 **kwargs):
        super().__init__(encoder_name=encoder_name,
                         in_channels=in_channels,
                         classes=classes, **kwargs)
        self.segmentation_head[0].bias.data.fill_(output_init_bias)


if __name__ == '__main__':
    T = AdversarialMaskTransform(32, [3, 224, 224])
    x = torch.randn(32, 3, 224, 224)
    T_x = T(x)
