import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
from torchvision.transforms.transforms import RandomCrop
from torchvision.datasets import VisionDataset, CIFAR10
from torchvision.datasets.folder import is_image_file
import os
import numpy as np


class CCSE(VisionDataset):
    def __init__(self, transform=None, label_transform=None,
                 sample_csv='/home/labuser/Datasets/cc12m/500K.csv',
                 image_dir='/home/labuser/Datasets/cc12m/training',
                 embedding_dir='/home/labuser/Datasets/cc12m/sentence_embedding_mpnet_base',
                 embedding_file='/home/labuser/Datasets/cc12m/all_embeddings.npy'):
        self.transform = transform
        self.label_transform = label_transform
        self.sample_csv = sample_csv
        self.image_dir = image_dir
        self.embedding_dir = embedding_dir
        self.embedding_file = embedding_file
        if self.embedding_file is not None and os.path.isfile(self.embedding_file):
            self.embeddings = np.load(embedding_file, mmap_mode='r')
        else:
            self.embeddings = None
        self.samples_idxs = pd.read_csv(self.sample_csv, header=None)[0].tolist()

    def __len__(self) -> int:
        return len(self.samples_idxs)

    def __getitem__(self, i):
        index = self.samples_idxs[i]
        image = Image.open(os.path.join(self.image_dir, f'{index:08d}.jpg'))
        image = image.convert('RGB')
        if self.embeddings is None:
            embedding = torch.load(os.path.join(self.embedding_dir, f'{index:08d}.pt'))
        else:
            embedding = torch.from_numpy(self.embeddings[index])
        if self.transform is not None:
            image = self.transform(image)
        if self.label_transform is not None:
            embedding = self.label_transform(embedding)
        return image, embedding


class CC12M(Dataset):
    def __init__(self, transform=None, label_transform=None,
                 tsv_path='/home/labuser/Datasets/cc12m/cc12m.tsv',
                 path='/home/labuser/Datasets/cc12m/training'):
        self.path = path
        self.transform = transform
        self.label_transform = label_transform

        self.data = pd.read_table(tsv_path, index_col=False,
                                  names=['url', 'caption'], usecols=['caption'])

        # self.data = next(iter(data_iter))

    def image_filename(self, i):
        return f'{self.path}/{i:08d}.jpg'

    def show_image(self, num):
        img = Image.open(self.image_filename(num))
        print(self.data['caption'][num])
        img.show()

    def __getitem__(self, index):
        caption = self.data['caption'][index]
        img = Image.open(self.image_filename(index))
        if self.transform is not None:
            img = self.transform(img)
        if self.label_transform is not None:
            caption = self.label_transform(caption)
        return img, caption


class CC5Ms(Dataset):
    def __init__(self, transform=None, label_transform=None,
                 split='train',
                 path='/home/labuser/Datasets/cc12m/',
                 training_filenames=('cc12m_small_5M_images.npy', 'cc12m_small_5M_embeddings.npy'),
                 eval_filenames=('cc12m_small_5K_images.npy', 'cc12m_small_5K_embeddings.npy')):
        if split == 'train':
            self.filenames = training_filenames
        elif split == 'eval':
            self.filenames = eval_filenames
        self.data = np.load(os.path.join(path, self.filenames[0]), mmap_mode='r')
        self.embeddings = np.load(os.path.join(path, self.filenames[1]), mmap_mode='r')
        self.transform = transform
        self.label_transform = label_transform
        assert self.data.shape[0] == self.embeddings.shape[0]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        embedding = torch.from_numpy(np.copy(self.embeddings[index]))
        if self.transform is not None:
            img = self.transform(img)
        if self.label_transform is not None:
            embedding = self.label_transform(embedding)
        return img, embedding


class NormalizeEmbeddings:
    def __init__(self, mean_filename, std_filename, eps=1e-8) -> None:
        self.mean = torch.from_numpy(np.load(mean_filename))
        self.std = torch.from_numpy(np.load(std_filename))
        self.eps = eps

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__ + '()'
