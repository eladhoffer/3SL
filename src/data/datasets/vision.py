import os
from torchvision import datasets
from torch.utils.data import Subset
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def vision_datasets(name='cifar10', split='train', transform=None,
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
