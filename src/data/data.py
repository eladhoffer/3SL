import os
import torch
from torchvision import datasets
from torchvision.transforms import transforms
from typing import Optional, Tuple
from pytorch_lightning import LightningDataModule
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset
from torch._utils import _accumulate
from src.utils_pt.regime import Regime
from src.utils_pt.dataset import IndexedFileDataset
from itertools import chain
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def get_dataset(name, split='train',
                transform=None, target_transform=None,
                download=False, path='~/Datasets', **kwargs):
    train = (split == 'train')
    root = os.path.join(os.path.expanduser(path), name)
    if name == 'cifar10':
        return datasets.CIFAR10(root=root,
                                train=train,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)
    elif name == 'cifar100':
        return datasets.CIFAR100(root=root,
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'mnist':
        return datasets.MNIST(root=root,
                              train=train,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'stl10':
        return datasets.STL10(root=root,
                              split=split,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'imagenet':
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        return datasets.ImageFolder(root=root,
                                    transform=transform,
                                    target_transform=target_transform)
    elif name == 'imagenet_tar':
        if train:
            root = os.path.join(root, 'imagenet_train.tar')
        else:
            root = os.path.join(root, 'imagenet_validation.tar')
        return IndexedFileDataset(root, extract_target_fn=(
            lambda fname: fname.split('/')[0]),
            transform=transform,
            target_transform=target_transform)


class DataRegime(object):
    def __init__(self, regime, defaults={}):
        self.regime = Regime(regime, deepcopy(defaults))
        self.epoch = 0
        self.steps = None
        self.get_loader(True)

    def get_setting(self):
        setting = {**self.regime.setting}
        subset_indices = setting.get('subset_indices', None)
        loader_setting = setting.get('loader', {})
        transform = setting.get('transform', None)
        return {'data': setting,
                'loader': loader_setting,
                'transform': transform,
                'subset_indices': subset_indices}

    def get(self, key, default=None):
        return self.regime.setting.get(key, default)

    def get_loader(self, force_update=False, override_settings=None):
        if force_update or self.regime.update(self.epoch, self.steps):
            setting = self.get_setting()
            if override_settings is not None:
                setting.update(override_settings)
            duplicates = setting.get('duplicates', 1)
            self._transform = setting['transform']
            setting['data'].setdefault('transform', self._transform)
            self._data = get_dataset(**setting['data'])
            if setting['subset_indices'] is not None:
                self._data = Subset(self._data, setting['subset_indices'])
            if setting['loader'].get('distributed', False):
                setting['loader']['sampler'] = DistributedSampler(self._data)
                setting['loader']['shuffle'] = None
                # pin-memory currently broken for distributed
                setting['loader']['pin_memory'] = False
            self._sampler = setting['loader'].get('sampler', None)
            self._loader = torch.utils.data.DataLoader(
                self._data, **setting['loader'])
        return self._loader

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self._sampler is not None and hasattr(self._sampler, 'set_epoch'):
            self._sampler.set_epoch(epoch)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return str(self.regime)


class DataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        **configs,
    ):
        super().__init__()
        self.configs = configs
        self.datasets = {}

    @property
    def num_classes(self) -> int:
        return None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        for config in self.configs.values():
            config = deepcopy(config)
            config['download'] = True
            DataRegime(None, defaults=config)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        for name, config in self.configs.items():
            self.datasets[name] = DataRegime(None, defaults=config)

    def get_dataloader(self, name):
        dataset = self.datasets.get(name, None)
        if dataset is None:
            return None
        else:
            return dataset.get_loader()

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('val')

    def test_dataloader(self):
        return self.get_dataloader('test')


# def _get_dataset(name, split='train', transform=None,
#                  target_transform=None, download=True, path='~/Datasets'):
#     train = (split == 'train')
#     root = os.path.join(os.path.expanduser(path), name)
#     if name == 'cifar10':
#         dataset = datasets.CIFAR10(root=root,
#                                    train=train,
#                                    transform=transform,
#                                    target_transform=target_transform,
#                                    download=download)
#         num_classes = 10
#         sample_size = (3, 32, 32)
#     elif name == 'cifar100':
#         dataset = datasets.CIFAR100(root=root,
#                                     train=train,
#                                     transform=transform,
#                                     target_transform=target_transform,
#                                     download=download)
#         num_classes = 10
#         sample_size = (3, 32, 32)

#     elif name == 'mnist':
#         dataset = datasets.MNIST(root=root,
#                                  train=train,
#                                  transform=transform,
#                                  target_transform=target_transform,
#                                  download=download)
#         num_classes = 10
#         sample_size = (1, 28, 28)
#     elif name == 'stl10':
#         dataset = datasets.STL10(root=root,
#                                  split=split,
#                                  transform=transform,
#                                  target_transform=target_transform,
#                                  download=download)
#         num_classes = 10
#         sample_size = (3, 96, 96)
#     elif name == 'imagenet':
#         if train:
#             root = os.path.join(root, 'train')
#         else:
#             root = os.path.join(root, 'val')
#         dataset = datasets.ImageFolder(root=root,
#                                        transform=transform,
#                                        target_transform=target_transform)
#         num_classes = 1000
#         sample_size = (3, None, None)

#     return {
#         'dataset': dataset,
#         'num_classes': num_classes,
#         'sample_size': sample_size
#     }
