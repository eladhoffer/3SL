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


def get_dataset(name='cifar10', split='train', transform=None,
                target_transform=None, download=True, path='~/Datasets',
                subset_indices=None, **kwargs):
    train = (split == 'train')
    root = os.path.join(os.path.expanduser(path), name)
    if name == 'cifar10':
        dataset = datasets.CIFAR10(root=root,
                                   train=train,
                                   transform=transform,
                                   target_transform=target_transform,
                                   download=download)
        num_classes = 10
        sample_size = (3, 32, 32)
    elif name == 'cifar100':
        dataset = datasets.CIFAR100(root=root,
                                    train=train,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=download)
        num_classes = 10
        sample_size = (3, 32, 32)

    elif name == 'mnist':
        dataset = datasets.MNIST(root=root,
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
        num_classes = 10
        sample_size = (1, 28, 28)
    elif name == 'stl10':
        dataset = datasets.STL10(root=root,
                                 split=split,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
        num_classes = 10
        sample_size = (3, 96, 96)
    elif name == 'imagenet':
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        dataset = datasets.ImageFolder(root=root,
                                       transform=transform,
                                       target_transform=target_transform)
        num_classes = 1000
        sample_size = (3, None, None)

    if subset_indices is not None:
        dataset = Subset(dataset, subset_indices)
    return {
        'dataset': dataset,
        'num_classes': num_classes,
        'sample_size': sample_size,
        **kwargs
    }


def _update(x, default={}, override={}):
    if isinstance(x, dict):
        for key, value in default.items():
            if key not in x.keys():
                x[key] = value
            elif isinstance(value, dict):
                _update(x[key], value)
        for key, value in override.items():
            if isinstance(value, dict):
                x.setdefault(key, {})
                _update(x[key], override=value)
            else:
                x[key] = value
    return x


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
        train=None,
        val=None,
        test=None,
        benchmark=None,
        num_workers=None,  # override for ddp usage
        pin_memory=None,  # override for ddp usage,
        **kwargs
    ):
        super().__init__()
        defaults = {}
        override = {'loader': {}}
        if num_workers is not None:
            override['loader']['num_workers'] = num_workers
        if pin_memory is not None:
            override['loader']['pin_memory'] = pin_memory
        self.configs = {
            'train': _update(train, defaults, override),
            'val': _update(val, defaults, override),
            'test': _update(test, defaults, override),
            'benchmark': _update(benchmark, defaults, override)
        }
        self.datasets = {}

    @property
    def num_classes(self) -> int:
        return None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""

        for config in self.configs.values():
            if config is None:
                continue
            dataset = config.get('dataset', None)
            if isinstance(dataset, dict):
                dataset = deepcopy(dataset)
                dataset['download'] = True
                get_dataset(**dataset)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        for name, config in self.configs.items():
            if config is None:
                continue
            dataset = config.get('dataset', None)
            if isinstance(dataset, dict):
                dataset = get_dataset(**dataset)['dataset']
            self.datasets[name] = dataset

    def get_dataloader(self, name):
        dataset = self.datasets.get(name, None)
        if dataset is None:
            return None
        else:
            loader_config = self.configs[name]['loader']
            return DataLoader(dataset, **loader_config)

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('val')

    def test_dataloader(self):
        return self.get_dataloader('test')
