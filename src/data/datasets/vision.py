import os
from torchvision import datasets
from torch.utils.data import Subset
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

def split_label_phases(dataset, phase_classes=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]):
    idxs = [[] for _ in range(len(phase_classes))]
    for i, (_, label) in enumerate(dataset):
        for j, classes in enumerate(phase_classes):
            if label in classes:
                idxs[j].append(i)

    return [Subset(dataset, indices) for indices in idxs]

def select_by_class(dataset, classes):
    idxs = []
    for i, (_, label) in enumerate(dataset):
        if label in classes:
            idxs.append(i)
    return Subset(dataset, idxs)

def vision_datasets(name='cifar10', split='train', transform=None,
                    target_transform=None, download=False, path='~/Datasets',
                    subset_indices=None, subset_classes=None, **kwargs):
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
    if subset_classes is not None:
        dataset = select_by_class(dataset, subset_classes)
        num_classes = len(subset_classes)
    return {
        'dataset': dataset,
        'num_classes': num_classes,
        'sample_size': sample_size,
        **kwargs
    }
