# from .augment import Duplicate, _TRANSFORMS
from torchvision.transforms import *
from .utils import *
from .autoaugment import ImageNetPolicy as AutoAugmentImageNet
from .autoaugment import CIFAR10Policy as AutoAugmentCifar
from .randaugment import RandAugmentMC
import torch
import numpy as np


class NormalizeEmbeddings:
    def __init__(self, mean_filename, std_filename, eps=1e-8) -> None:
        self.mean = torch.from_numpy(np.load(mean_filename))
        self.std = torch.from_numpy(np.load(std_filename))
        self.eps = eps

    def __call__(self, tensor):
        return (tensor - self.mean) / (self.std + self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '()'
