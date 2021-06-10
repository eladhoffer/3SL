# from .augment import Duplicate, _TRANSFORMS
from torchvision.transforms import *
from .utils import *
from .autoaugment import ImageNetPolicy as AutoAugmentImageNet
from .autoaugment import CIFAR10Policy as AutoAugmentCifar
from .randaugment import RandAugmentMC
