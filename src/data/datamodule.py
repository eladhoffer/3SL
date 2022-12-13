import os
import torch
from torchvision import datasets
from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from hydra.utils import instantiate
from omegaconf import OmegaConf
from copy import deepcopy
from src.data.datasets.vision import vision_datasets
from src.utils_pt.regime import Regime


class DataConfig:
    def __init__(self, config, current_epoch=0):
        self._config = config
        self.current_epoch = current_epoch

    @property
    def config(self):
        if isinstance(self._config, Regime):
            self._config.update(epoch=self.current_epoch)
            merged_config = OmegaConf.merge(self._config.defaults, self._config.setting)
            return merged_config
        else:
            return self._config

    @staticmethod
    def _get_dataset(dataset, **kwargs):
        try:
            dataset = instantiate(dataset, _convert_="all")
        except:
            pass
        if isinstance(dataset, dict):  # if dict, assume it's a vision dataset config
            dataset = deepcopy(dataset)
            dataset.update(kwargs)
            dataset = vision_datasets(**dataset)['dataset']
        return dataset

    @staticmethod
    def _get_loader(loader, dataset):
        loader = instantiate(loader, dataset=dataset, _convert_="all")
        if isinstance(loader, dict):
            loader = DataLoader(**loader)
        return loader

    @staticmethod
    def _extract_datasets(config, **kwargs):
        if config is None:
            return None
        dataset = config.get('dataset', None)
        if dataset is not None:
            dataset = DataConfig._get_dataset(dataset, **kwargs)
            transform = config.get('transform', None)
            if transform is not None:
                dataset = transform(dataset)
            return dataset

        datasets = {key: DataConfig._extract_datasets(value, **kwargs)
                    for key, value in config.items() if value is not None}
        if len(datasets) == 0:
            return None
        return datasets

    @staticmethod
    def _extract_loaders(config, **kwargs):
        if config is None:
            return None
        loader = config.get('loader', None)
        if loader is not None:
            dataset = DataConfig._extract_datasets(config, **kwargs)
            if dataset is None:
                return None
            return DataConfig._get_loader(loader, dataset)
        return {key: DataConfig._extract_loaders(value, **kwargs)
                for key, value in config.items() if value is not None}

    def dataset(self, download=False):       
        return self._extract_datasets(self.config, download=download)

    def loader(self):
        return self._extract_loaders(self.config)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    @staticmethod
    def from_regime(regime, **defaults):
        """ regime is a list of dicts with keys 'epochs' configuration"""
        return DataConfig(Regime(regime, defaults=defaults))


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

        def _get_config(config):
            if config is None:
                return None
            if hasattr(config, '_target_'):
                config = instantiate(config, _convert_="all", _recursive_=False)
            if isinstance(config, DataConfig):
                return config
            return DataConfig(config)
        self.configs = {
            'train': _get_config(train),
            'val': _get_config(val),
            'test': _get_config(test),
            'benchmark': _get_config(benchmark),
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
            config.set_epoch(self.trainer.current_epoch)
            return config.loader()

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('val')

    def test_dataloader(self):
        return self.get_dataloader('test')
