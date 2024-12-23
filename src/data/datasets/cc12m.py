import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
from torchvision.transforms.transforms import RandomCrop
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import is_image_file, default_loader
import os
import numpy as np
from datasets import Dataset as dset


class CCSE(VisionDataset):
    def __init__(self,
                 image_dir,
                 sample_csv,
                 embedding_dir=None,
                 embedding_file=None,
                 transform=None, label_transform=None):
        self.transform = transform
        self.label_transform = label_transform
        self.sample_csv = sample_csv
        self.image_dir = image_dir
        self.embedding_dir = embedding_dir
        self.embedding_file = embedding_file
        if self.embedding_file is not None:
            assert os.path.isfile(self.embedding_file)
            # self.embeddings = np.load(embedding_file, mmap_mode='r')
        else:
            assert os.path.isdir(self.embedding_dir)
        self.samples_idxs = torch.tensor(pd.read_csv(
            self.sample_csv, header=None)[0].tolist())

    def __len__(self) -> int:
        return len(self.samples_idxs)

    def __getitem__(self, i):
        index = self.samples_idxs[i]
        image = default_loader(os.path.join(
            self.image_dir, f'{index:08d}.jpg'))
        if self.embedding_file:
            embedding = np.load(self.embedding_file, mmap_mode='r')[index]
            embedding = torch.from_numpy(embedding)
        else:
            embedding = torch.load(os.path.join(
                self.embedding_dir, f'{index:08d}.pt'))
        # if self.embeddings is None:
        #     embedding = torch.load(os.path.join(self.embedding_dir, f'{index:08d}.pt'))
        # else:
        #     embedding = torch.from_numpy(self.embeddings[index])
        if self.transform is not None:
            image = self.transform(image)
        if self.label_transform is not None:
            embedding = self.label_transform(embedding)
        return image, embedding


class CC12M(Dataset):
    def __init__(self, transform=None, label_transform=None,
                 tsv_path='cc12m.tsv',
                 sample_csv='10M.csv',
                 path='training',
                 return_dict=True):
        self.path = path
        self.transform = transform
        self.label_transform = label_transform
        self.sample_csv = sample_csv
        self.return_dict = return_dict
        if sample_csv is not None:
            self.samples_idxs = pd.read_csv(
                self.sample_csv, header=None)[0].tolist()

        self.data = pd.read_table(tsv_path, index_col=False,
                                  names=['url', 'caption'], usecols=['caption'])

    def __len__(self) -> int:
        return len(self.data) if self.samples_idxs is None \
            else len(self.samples_idxs)

    def image_filename(self, i):
        return f'{self.path}/{i:08d}.jpg'

    def show_image(self, num):
        img = Image.open(self.image_filename(num))
        img.show()

    def __getitem__(self, index):
        if self.samples_idxs is not None:
            index = self.samples_idxs[index]
        caption = self.data['caption'][index]
        img = default_loader(self.image_filename(index))
        if self.transform is not None:
            img = self.transform(img)
        if self.label_transform is not None:
            caption = self.label_transform(caption)
        if self.return_dict:
            return {'image': img, 'text': caption}
        else:
            return img, caption


class CC12MTokenized(CC12M):
    def __init__(self, tokenizer, transform=None, label_transform=None,
                 tsv_path='cc12m.tsv',
                 sample_csv='10M.csv',
                 path='training',
                 max_length=128,
                 cache_dir='tokenized'):
        cache_file = f"{cache_dir}/cc12m_tokenized.arrow"
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        self.label_transform = label_transform
        self.path = path
        tokenizer_args = {"truncation": True,
                          "max_length": max_length}

        def _tokenize(sample, idx):
            output = self.tokenizer(sample['text'].split('\t')[
                                    1], **tokenizer_args)
            output['idx'] = idx
            return output
        if os.path.isfile(cache_file):
            self.data = dset.from_file(cache_file)
        else:
            self.data = dset.from_text(tsv_path)
            self.data = self.data.map(_tokenize, remove_columns=['text'],
                                      with_indices=True, num_proc=32,
                                      cache_file_name=cache_file)
        self.data.set_format(type='torch', columns=[
                             'idx', 'input_ids', 'attention_mask'])
        if sample_csv is not None:
            samples_idxs = pd.read_csv(sample_csv, header=None)[0].tolist()
            self.data = self.data.select(samples_idxs)

    def __getitem__(self, index):
        sample = self.data[index]
        caption = {'input_ids': sample['input_ids'],
                   'attention_mask': sample['attention_mask']}
        img_idx = sample['idx']
        img = default_loader(self.image_filename(img_idx))
        if self.transform is not None:
            img = self.transform(img)
        if self.label_transform is not None:
            caption = self.label_transform(caption)
        return {'image': img, 'text': caption}

    def __len__(self):
        return len(self.data)


class CC5Ms(Dataset):
    def __init__(self, path, transform=None, label_transform=None,
                 split='train',
                 training_filenames=(
                     'cc12m_small_5M_images.npy', 'cc12m_small_5M_embeddings.npy'),
                 eval_filenames=('cc12m_small_5K_images.npy', 'cc12m_small_5K_embeddings.npy')):
        if split == 'train':
            self.filenames = training_filenames
        elif split == 'eval':
            self.filenames = eval_filenames
        self.data = np.load(os.path.join(
            path, self.filenames[0]), mmap_mode='r')
        self.embeddings = np.load(os.path.join(
            path, self.filenames[1]), mmap_mode='r')
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


class CC10Ms(CC5Ms):
    def __init__(self, path, transform=None, label_transform=None,
                 split='train',
                 training_filenames=('cc12m_small_10M_images.npy',
                                     'cc12m_small_10M_embeddings.npy'),
                 eval_filenames=('cc12m_small_5K_images.npy', 'cc12m_small_5K_embeddings.npy')):
        super().__init__(transform, label_transform, split,
                         path, training_filenames, eval_filenames)


class NormalizeEmbeddings:
    def __init__(self, mean_filename, std_filename, eps=1e-8) -> None:
        self.mean = torch.from_numpy(np.load(mean_filename))
        self.std = torch.from_numpy(np.load(std_filename))
        self.eps = eps

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__ + '()'
