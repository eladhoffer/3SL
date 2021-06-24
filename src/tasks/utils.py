
import torch
import torch.nn as nn


def pack_inputs(*tensors, item_num_dims=3):
    shapes = []
    tensor_list = []
    for t in tensors:
        s = t.shape[:-item_num_dims]
        if len(s) > 1:  # batch is over two dims
            t = t.view(-1, *t.shape[len(s):])
        tensor_list.append(t)
        shapes.append(s)
    return torch.cat(tensor_list), shapes


def unpack_outputs(tensor, shapes):
    idx = 0
    out = []
    for shape in shapes:
        if len(shape) == 1:
            B = shape[0]
        else:
            B = torch.tensor(shape).prod()
        out_shape = shape + tensor.shape[1:]
        out.append(tensor[idx:idx + B].view(*out_shape))
        idx += B
    return tuple(out)


def mlp_normalized(feats, final_act=False, final_norm=False,
                   act=nn.ReLU, act_kwargs={'inplace': True},
                   norm=nn.BatchNorm1d, norm_kwargs={}):
    modules = []
    for i in range(1, len(feats)):
        if not final_act and i == len(feats) - 1:
            act_layer = None
        else:
            act_layer = act(**act_kwargs)
        if not final_norm and i == len(feats) - 1:
            norm_layer = None
        else:
            norm_layer = norm(feats[i], **norm_kwargs)

        modules.append(nn.Linear(feats[i - 1], feats[i],
                                 bias=norm_layer is None),
                       )
        if norm_layer is not None:
            modules.append(norm_layer)
        if act_layer is not None:
            modules.append(act_layer)

    return nn.Sequential(*modules)


class DiffImage(nn.Module):
    def __init__(self, num_classes, image_size,
                 num_channels=3, num_images=1, augment=None, act=nn.Identity()):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_images = num_images
        self.act = act
        self.embedding = nn.Embedding(num_classes * num_images,
                                      num_channels * image_size * image_size)

        self.augment = augment
        self.bn = nn.BatchNorm2d(num_channels)
        self.init()

    def init(self, clamped=False):
        with torch.no_grad():
            self.embedding.weight.uniform_()
            if clamped:
                self.embedding.weight.add_(-self.embedding.weight.min())
                self.embedding.weight.div_(-self.embedding.weight.max())

    def get_images(self):
        return self.embedding.weight.view(-1, self.num_channels, self.image_size, self.image_size)

    def image_loader(self):
        idxs = torch.arange(self.embedding.weight.size(0))
        imgs = self.embedding.weight.view(-1, self.num_channels,
                                          self.image_size, self.image_size)
        imgs = self.augment(imgs) if self.augment is not None else imgs
        imgs = self.bn(imgs)
        labels = idxs % self.num_classes
        return [(imgs, labels)]

    def random_entry(self, labels):
        random_offset = self.num_classes \
            * torch.randint(self.num_images, labels.shape,
                            device=labels.device)
        return labels + random_offset

    def add_images(self, images, labels, denormalize=False, normalization=_imagenet_stats):
        std = normalization['std'].view(-1, 1, 1)
        mean = normalization['mean'].view(-1, 1, 1)
        if self.num_images > 1:
            labels = self.random_entry(labels)
        with torch.no_grad():
            for img, lbl in zip(images, labels):
                if denormalize:
                    img = img * std + mean
                self.embedding.weight[lbl].copy_(img.view(-1))

    def forward(self, label):
        if self.num_images > 1:
            label = self.random_entry(label)
        x = self.embedding(label)
        x = self.act(x)
        x = x.view(-1, self.num_channels, self.image_size, self.image_size)
        if self.augment is not None:
            x = self.augment(x)
        x = self.bn(x)
        return x
