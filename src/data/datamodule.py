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


class DataConfig:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def _get_dataset(dataset, **kwargs):
        if isinstance(dataset, dict):
            dataset = deepcopy(dataset)
            dataset.update(kwargs)
            dataset = DataConfig._vision_datasets(**dataset)['dataset']
        return dataset

    @staticmethod
    def _extract_datasets(config, **kwargs):
        if config is None:
            return None
        dataset = config.get('dataset', None)
        if dataset is not None:
            return DataConfig._get_dataset(dataset, **kwargs)
        return {key: DataConfig._extract_datasets(value, **kwargs)
                for key, value in config.items() if value is not None}

    @staticmethod
    def _extract_loaders(config, **kwargs):
        if config is None:
            return None
        dataset = config.get('dataset', None)
        loader = config.get('loader', None)
        if loader is not None:
            dataset = DataConfig._get_dataset(dataset, **kwargs)
            return DataLoader(dataset, **loader)
        return {key: DataConfig._extract_loaders(value, **kwargs)
                for key, value in config.items() if value is not None}

    def dataset(self, download=False):
        return self._extract_datasets(self.config, download=download)

    def loader(self):
        return self._extract_loaders(self.config)

    @staticmethod
    def _vision_datasets(name='cifar10', split='train', transform=None,
                         target_transform=None, download=False, path='~/Datasets',
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
        **kwargs
    ):
        super().__init__()
        self.configs = {
            'train': DataConfig(train) if train else None,
            'val': DataConfig(val) if val else None,
            'test': DataConfig(test) if test else None,
            'benchmark': DataConfig(benchmark) if benchmark else None
        }

    @property
    def num_classes(self) -> int:
        return None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""

        for config in self.configs.values():
            if config is None:
                continue
            config.dataset(download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        pass

    def get_dataloader(self, name):
        config = self.configs.get(name, None)
        if config is None:
            return None
        else:
            return config.loader()

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('val')

    def test_dataloader(self):
        return self.get_dataloader('test')
