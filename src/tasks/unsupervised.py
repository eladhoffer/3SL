from lightly.transforms import gaussian_blur
from .task import pack_inputs, unpack_outputs, MixUp
from .supervised import ClassificationTask
from pytorch_lightning.metrics import functional as FM
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.cross_entropy import cross_entropy
import lightly
from lightly.utils.benchmarking import knn_predict
from itertools import combinations
import torch.distributed as dist
from data.augment import DiffSimCLRAugment, _imagenet_stats
import torchvision


class UnSupTask(ClassificationTask):

    def __init__(self, model, regime, **kwargs):
        super().__init__(model, regime, **kwargs)

    def unsup_loss(self, outputs, _real_target=None):
        pass

    def training_step(self, batch, batch_idx):
        self.update_regime()
        unlabeled, _ = batch
        inputs, shapes = pack_inputs(unlabeled)
        output = self.model(inputs)
        output = unpack_outputs(output, shapes)
        loss = self.unsup_loss(output)
        self.log('lr', self.optimizer.get_lr()[0], on_step=True)
        self.log_dict({'loss/train': loss},
                      prog_bar=True, on_epoch=True, on_step=True)
        return loss


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
                 num_channels=3, num_images=1, augment=None):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_images = num_images
        self.embedding = nn.Embedding(num_classes * num_images,
                                      num_channels * image_size * image_size)
        with torch.no_grad():
            self.embedding.weight.uniform_()
            # self.embedding.weight.add_(-self.embedding.weight.min())
            # self.embedding.weight.div_(-self.embedding.weight.max())

        self.augment = augment
        self.bn = nn.BatchNorm2d(num_channels)

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
        # x = x.sigmoid()  # image data is in [0,1]
        # x = x.clamp(min=0., max=1.)
        x = x.view(-1, self.num_channels, self.image_size, self.image_size)
        if self.augment is not None:
            x = self.augment(x)
        x = self.bn(x)
        return x


class SimSiamTask(UnSupTask):

    def __init__(self, model, regime,
                 num_feats=512, proj_feats=[2048, 2048], pred_feats=[512],
                 num_classes=10, class_embedding='features', num_image_aug=1,
                 loss_type='cosine',
                 **kwargs):
        # create a simsiam model based on ResNet
        # model = \
        #     lightly.models.SimSiam(
        #         model, num_ftrs=512, num_mlp_layers=2)
        self.class_embedding = class_embedding
        if class_embedding == 'features':
            embedding = nn.Embedding(num_classes, proj_feats[-1])
        elif class_embedding == 'proj_features':
            embedding = nn.Embedding(num_classes, num_feats)
        elif 'image' in class_embedding:
            augment = None
            if 'augmented_image' in class_embedding:
                augment = DiffSimCLRAugment(input_size=32, gaussian_blur=0.)
            embedding = DiffImage(num_classes, image_size=32,
                                  num_images=num_image_aug, augment=augment)
        else:
            embedding = None

        model = nn.ModuleDict({
            'backbone': model,
            'projection': mlp_normalized([num_feats] + proj_feats, final_norm=True),
            # 'projection': nn.Identity(),
            # 'prediction': mlp_normalized([proj_feats[-1]] + pred_feats + [proj_feats[-1]]),
            'embedding': embedding
        })
        super().__init__(model, regime, **kwargs)
        if loss_type == 'cosine':
            self.criterion = lightly.loss.SymNegCosineSimilarityLoss()
        elif loss_type == 'barlow':
            self.criterion = lightly.loss.BarlowTwinsLoss()

    def unsup_loss(self, output_views, _real_target=None):
        v1, v2 = output_views
        return self.criterion(v1, v2)
        # loss = 0
        # for v1, v2 in combinations(output_views, 2):
        #     loss = loss \
        #         - F.cosine_similarity(v1, v2.detach()).mean() / 2 \
        #         - F.cosine_similarity(v1.detach(), v2).mean() / 2

        # return loss
    # def simsiam_step(self, x0, x1=None, label=None, forward_together=True):
    #     backbone = self.model['backbone']
    #     proj = self.model['projection']
    #     predict = self.model['prediction']
    #     if forward_together:
    #         if label is not None:
    #             if 'image' in self.class_embedding:
    #                 x1 = self.model['embedding'](label)
    #         if x1 is not None:
    #             x, shape = pack_inputs(x0, x1)
    #             z = proj(backbone(x).flatten(1, -1))
    #             h = predict(z)
    #             h0, h1 = unpack_outputs(h, shape)
    #             z0, z1 = unpack_outputs(z, shape)
    #     else:
    #         if x1 is not None:
    #             z1 = proj(backbone(x1).flatten(1, -1))
    #             h1 = predict(z1)
    #         elif label is not None:
    #             if self.class_embedding == 'features':
    #                 z1 = self.model['embedding'](label)
    #             elif self.class_embedding == 'proj_features':
    #                 z1 = proj(self.model['embedding'](label))
    #             elif 'image' in self.class_embedding:
    #                 if 'init' in self.class_embedding:
    #                     self.model['embedding'].add_images(x0, label)
    #                     x0 = self.model['embedding'].transform(x0)
    #                 x1 = self.model['embedding'](label)
    #                 z1 = proj(backbone(x1).flatten(1, -1))
    #             h1 = predict(z1)
    #         z0 = proj(backbone(x0).flatten(1, -1))
    #         h0 = predict(z0)
    #     loss = self.unsup_loss(((z0, h0), (z1, h1)))
    #     return loss

    def simsiam_step(self, x0, x1):
        backbone = self.model['backbone']
        proj = self.model['projection']
        # predict = self.model['prediction']
        z0 = proj(backbone(x0).flatten(1, -1))
        # h0 = predict(z0)
        z1 = proj(backbone(x1).flatten(1, -1))
        # h1 = predict(z1)
        # loss = self.unsup_loss(((z0, h0), (z1, h1)))
        loss = self.criterion(z0, z1)
        return loss

    def training_step(self, batch, batch_idx):
        self.update_regime()

        (xl, label), (x, _) = batch
        x0, x1 = x.unbind(dim=1)
        if label is not None and 'image' in self.class_embedding:
            x0 = torch.cat((x0, xl))
            x1 = torch.cat((x1, self.model['embedding'](label)))
        loss = self.simsiam_step(x0, x1)
        # if xl is not None and label is not None:
        # loss = (loss + self.simsiam_step(xl, label=label)) / 2.
        self.log('loss/train', loss)
        self.log('lr', self.optimizer.get_lr()[0], on_step=True, prog_bar=True)
        return loss

    def set_benchmark(self, name='', **config):
        self.benchmarks = getattr(self, 'benchmarks', {})
        self.benchmarks[name] = config

    def update_feature_bank(self):
        if not hasattr(self, 'benchmarks'):
            return
        self.eval()
        backbone = self.model['backbone']
        with torch.no_grad():
            for benchmark in self.benchmarks.values():
                feature_bank_device = benchmark['feature_bank_device']
                benchmark['feature_bank'] = []
                benchmark['targets_bank'] = []
                dataloader = benchmark['dataloader']
                if callable(dataloader):
                    dataloader = dataloader()
                for data in dataloader:
                    img, target = data
                    img = img.to(self.device)
                    feature = backbone(img).squeeze()
                    feature = F.normalize(feature, dim=1)
                    feature = feature.to(feature_bank_device)
                    target = target.to(feature_bank_device)
                    benchmark['feature_bank'].append(feature)
                    benchmark['targets_bank'].append(target)
                benchmark['feature_bank'] = torch.cat(
                    benchmark['feature_bank'], dim=0).t().contiguous()
                benchmark['targets_bank'] = torch.cat(
                    benchmark['targets_bank'], dim=0).t().contiguous()
        self.train()

    def training_epoch_end(self, outputs):
        # update feature bank at the end of each training epoch
        self.update_feature_bank()
        if 'image' in self.class_embedding:
            grid = torchvision.utils.make_grid(
                self.model['embedding'].get_images(), normalize=True, scale_each=True)
            self.logger.experiment[0].add_image(
                'images', grid, self.global_step)

    def validation_step(self, batch, batch_idx):
        # we can only do kNN predictions once we have a feature bank
        if not hasattr(self, 'benchmarks'):
            return
        images, targets = batch
        model = getattr(self, '_model_ema', self.model)
        model.eval()
        feature = model['backbone'](images).squeeze()
        feature = F.normalize(feature, dim=1)

        acc = []
        for name, benchmark in self.benchmarks.items():
            if 'feature_bank' not in benchmark.keys():
                continue
            feature_bank_device = benchmark['feature_bank_device']
            feature = feature.to(feature_bank_device)
            targets = targets.to(feature_bank_device)
            pred_labels = knn_predict(
                feature,
                benchmark['feature_bank'],
                benchmark['targets_bank'],
                benchmark['num_classes'],
                benchmark['knn_k'],
                benchmark['knn_t']
            )
            top1 = (pred_labels[:, 0] == targets).float().mean().to(self.device)
            if len(name) > 0:
                var_name = f'accuracy/val({name})'
            else:
                var_name = 'accuracy/val'
            self.log(var_name, top1, on_epoch=True, prog_bar=True)
            acc.append(top1)

        return acc

    # def validation_epoch_end(self, outputs):
    #     device = self.feature_bank_device
    #     if outputs:
    #         total_num = torch.Tensor([0]).to(device)
    #         total_top1 = torch.Tensor([0.]).to(device)
    #         for (num, top1) in outputs:
    #             total_num += num[0]
    #             total_top1 += top1

    #         if dist.is_initialized() and dist.get_world_size() > 1:
    #             dist.all_reduce(total_num)
    #             dist.all_reduce(total_top1)

    #         acc = float(total_top1.item() / total_num.item())
    #         self.log('accuracy/val', acc, prog_bar=True)
