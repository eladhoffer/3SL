import torch
import numpy as np

_imagenet_stats = {'mean': torch.tensor([0.485, 0.456, 0.406]),
                   'std': torch.tensor([0.229, 0.224, 0.225])}

try:
    from lightly.data.collate import SimCLRCollateFunction

    def SimCLRAugment(**kwargs):
        return SimCLRCollateFunction(**kwargs).transform
except ImportError:
    pass

try:
    import kornia.augmentation as K
    import torch.nn as nn

    class DiffAugment(nn.Module):
        def __init__(self, input_size: int = 64,
                     cj_prob: float = 0.8,
                     cj_bright: float = 0.7,
                     cj_contrast: float = 0.7,
                     cj_sat: float = 0.7,
                     cj_hue: float = 0.2,
                     min_scale: float = 0.15,
                     random_gray_scale: float = 0.2,
                     gaussian_blur: float = 0.5,
                     kernel_size: float = 0.1,
                     vf_prob: float = 0.0,
                     hf_prob: float = 0.5,
                     rr_prob: float = 0.0,
                     normalize: dict = {'mean': torch.tensor([0.485, 0.456, 0.406]),
                                        'std': torch.tensor([0.229, 0.224, 0.225])}):
            super().__init__()
            self.augment = nn.Sequential(
                K.RandomResizedCrop(size=(input_size, input_size),
                                    scale=(min_scale, 1.0)),
                K.RandomRotate(prob=rr_prob),
                K.RandomHorizontalFlip(p=hf_prob),
                K.RandomVerticalFlip(p=vf_prob),
                K.ColorJitter(cj_bright, cj_contrast,
                              cj_sat, cj_hue, p=cj_prob),
                K.RandomGrayscale(p=random_gray_scale),
                K.GaussianBlur(kernel_size=kernel_size * input_size,
                               prob=gaussian_blur),
                K.Normalize(**normalize)
            )
            self.normalize = K.Normalize(**normalize)

        def forward(self, x):
            if self.training:
                return self.augment(x)
            else:
                return self.normalize(x)

    class DiffSimCLRAugment(DiffAugment):
        def __init__(self,
                     input_size: int = 224,
                     cj_prob: float = 0.8,
                     cj_strength: float = 0.5,
                     min_scale: float = 0.08,
                     random_gray_scale: float = 0.2,
                     gaussian_blur: float = 0.5,
                     kernel_size: float = 0.1,
                     vf_prob: float = 0.0,
                     hf_prob: float = 0.5,
                     rr_prob: float = 0.0,
                     normalize={'mean': torch.tensor([0.485, 0.456, 0.406]),
                                'std': torch.tensor([0.229, 0.224, 0.225])}):
            super().__init__(
                input_size=input_size,
                cj_prob=cj_prob,
                cj_bright=cj_strength * 0.8,
                cj_contrast=cj_strength * 0.8,
                cj_sat=cj_strength * 0.8,
                cj_hue=cj_strength * 0.2,
                min_scale=min_scale,
                random_gray_scale=random_gray_scale,
                gaussian_blur=gaussian_blur,
                kernel_size=kernel_size,
                vf_prob=vf_prob,
                hf_prob=hf_prob,
                rr_prob=rr_prob,
                normalize=normalize,
            )
except ImportError:
    pass


class Stack(torch.nn.Module):
    """Stacks multiple transforms
    """

    def __init__(self, transforms, dim=0):
        super().__init__()
        self.transforms = transforms
        self.dim = dim

    def forward(self, img):
        return torch.stack([fn(img) for fn in self.transforms], dim=self.dim)


class Duplicate(torch.nn.Module):
    """preforms multiple transforms, useful to implement inference time augmentation or
     "batch augmentation" from https://openreview.net/forum?id=H1V4QhAqYQ&noteId=BylUSs_3Y7
    """

    def __init__(self, transform, num=1, dim=0):
        super().__init__()
        self.num = num
        self.transform = transform
        self.dim = dim

    def forward(self, img):
        return torch.stack([self.transform(img) for _ in range(self.num)], dim=self.dim)


class Cutout(object):
    """
    Randomly mask out one or more patches from an image.
    taken from https://github.com/uoguelph-mlrg/Cutout


    Args:
        holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, holes, length):
        self.holes = holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
