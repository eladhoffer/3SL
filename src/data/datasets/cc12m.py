import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.transforms import RandomCrop
from torchvision.datasets import VisionDataset, CIFAR10
from torchvision.datasets.folder import is_image_file
import os


class CCSE(VisionDataset):
    def __init__(self, transform=None, target_transform=None,
                 sample_csv='/home/labuser/Datasets/cc12m/500K.csv',
                 image_dir='/home/labuser/Datasets/cc12m/training',
                 embedding_dir='/home/labuser/Datasets/cc12m/sentence_embedding_mpnet_base'):
        self.transform = transform
        self.target_transform = target_transform
        self.sample_csv = sample_csv
        self.image_dir = image_dir
        self.embedding_dir = embedding_dir
        self.samples_idxs = pd.read_csv(self.sample_csv, header=None)[0].tolist()

    def __len__(self) -> int:
        return len(self.samples_idxs)

    def __getitem__(self, i):
        index = self.samples_idxs[i]
        image = Image.open(os.path.join(self.image_dir, f'{index:08d}.jpg'))
        image = image.convert('RGB')
        embedding = torch.load(os.path.join(self.embedding_dir, f'{index:08d}.pt'))
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)
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


# transform = transforms.Compose([
#     transforms.Resize(32),
#     transforms.CenterCrop(32),
#     transforms.ToTensor(),
# ])

# # dset = CC12M(transform=transform)
