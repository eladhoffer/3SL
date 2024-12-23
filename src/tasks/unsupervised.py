# from .utils import mlp_normalized, pack_inputs, unpack_outputs, DiffImage
from src.tasks.supervised import ClassificationTask
from torchmetrics import functional as FM
import torch
import torch.nn as nn
import torch.nn.functional as F
# import lightly
# from lightly.utils.benchmarking import knn_predict
# from src.data.transforms import DiffSimCLRAugment, _imagenet_stats
# import torchvision


# class UnSupTask(ClassificationTask):

#     def __init__(self, model, optimizer, **kwargs):
#         super().__init__(model, optimizer, **kwargs)

#     def unsup_loss(self, outputs, _real_target=None):
#         pass

#     def training_step(self, batch, batch_idx):
#         unlabeled, _ = batch
#         inputs, shapes = pack_inputs(unlabeled)
#         output = self.model(inputs)
#         output = unpack_outputs(output, shapes)
#         loss = self.unsup_loss(output)
#         self.log_lr(on_step=True)
#         self.log_dict({'loss/train': loss},
#                       prog_bar=True, on_epoch=True, on_step=True)
#         return loss


# class SimSiamTask(UnSupTask):

#     def __init__(self, model, regime,
#                  num_feats=512, proj_feats=[2048, 2048], pred_feats=[512],
#                  num_classes=10, class_embedding='features', num_image_aug=1,
#                  loss_type='cosine',
#                  **kwargs):
#         super().__init__(model, regime, **kwargs)

#         self.class_embedding = class_embedding
#         if class_embedding == 'features':
#             embedding = nn.Embedding(num_classes, proj_feats[-1])
#         elif class_embedding == 'proj_features':
#             embedding = nn.Embedding(num_classes, num_feats)
#         elif 'image' in class_embedding:
#             augment = None
#             if 'augmented_image' in class_embedding:
#                 augment = DiffSimCLRAugment(input_size=32)
#             embedding = DiffImage(num_classes, image_size=32,
#                                   num_images=num_image_aug, augment=augment)
#         else:
#             embedding = None

#         self.model.simsiam_aux = nn.ModuleDict({
#             'projection': mlp_normalized([num_feats] + proj_feats, final_norm=True),
#             'prediction': mlp_normalized([proj_feats[-1]] + pred_feats + [proj_feats[-1]]),
#             'embedding': embedding
#         })
#         if loss_type == 'cosine':
#             self.criterion = lightly.loss.SymNegCosineSimilarityLoss()
#         elif loss_type == 'barlow':
#             self.criterion = lightly.loss.BarlowTwinsLoss()

#     def unsup_loss(self, output_views, _real_target=None):
#         v1, v2 = output_views
#         return self.criterion(v1, v2)
#         # loss = 0
#         # for v1, v2 in combinations(output_views, 2):
#         #     loss = loss \
#         #         - F.cosine_similarity(v1, v2.detach()).mean() / 2 \
#         #         - F.cosine_similarity(v1.detach(), v2).mean() / 2

#         # return loss
#     # def simsiam_step(self, x0, x1=None, label=None, forward_together=True):
#     #     backbone = self.model['backbone']
#     #     proj = self.model['projection']
#     #     predict = self.model['prediction']
#     #     if forward_together:
#     #         if label is not None:
#     #             if 'image' in self.class_embedding:
#     #                 x1 = self.model['embedding'](label)
#     #         if x1 is not None:
#     #             x, shape = pack_inputs(x0, x1)
#     #             z = proj(backbone(x).flatten(1, -1))
#     #             h = predict(z)
#     #             h0, h1 = unpack_outputs(h, shape)
#     #             z0, z1 = unpack_outputs(z, shape)
#     #     else:
#     #         if x1 is not None:
#     #             z1 = proj(backbone(x1).flatten(1, -1))
#     #             h1 = predict(z1)
#     #         elif label is not None:
#     #             if self.class_embedding == 'features':
#     #                 z1 = self.model['embedding'](label)
#     #             elif self.class_embedding == 'proj_features':
#     #                 z1 = proj(self.model['embedding'](label))
#     #             elif 'image' in self.class_embedding:
#     #                 if 'init' in self.class_embedding:
#     #                     self.model['embedding'].add_images(x0, label)
#     #                     x0 = self.model['embedding'].transform(x0)
#     #                 x1 = self.model['embedding'](label)
#     #                 z1 = proj(backbone(x1).flatten(1, -1))
#     #             h1 = predict(z1)
#     #         z0 = proj(backbone(x0).flatten(1, -1))
#     #         h0 = predict(z0)
#     #     loss = self.unsup_loss(((z0, h0), (z1, h1)))
#     #     return loss

#     def simsiam_step(self, x0, x1):
#         backbone = self.model['backbone']
#         proj = self.model['projection']
#         # predict = self.model['prediction']
#         z0 = proj(backbone(x0).flatten(1, -1))
#         # h0 = predict(z0)
#         z1 = proj(backbone(x1).flatten(1, -1))
#         # h1 = predict(z1)
#         # loss = self.unsup_loss(((z0, h0), (z1, h1)))
#         loss = self.criterion(z0, z1)
#         return loss

#     def training_step(self, batch, batch_idx):
#         self.update_regime()

#         (xl, label), (x, _) = batch
#         x0, x1 = x.unbind(dim=1)
#         if label is not None and 'image' in self.class_embedding:
#             x0 = torch.cat((x0, xl))
#             x1 = torch.cat((x1, self.model['embedding'](label)))
#         loss = self.simsiam_step(x0, x1)
#         # if xl is not None and label is not None:
#         # loss = (loss + self.simsiam_step(xl, label=label)) / 2.
#         self.log('loss/train', loss)
#         self.log_lr(on_step=True)
#         return loss

#     def set_benchmark(self, name='', **config):
#         self.benchmarks = getattr(self, 'benchmarks', {})
#         self.benchmarks[name] = config

#     def update_feature_bank(self):
#         if not hasattr(self, 'benchmarks'):
#             return
#         self.eval()
#         backbone = self.model['backbone']
#         with torch.no_grad():
#             for benchmark in self.benchmarks.values():
#                 feature_bank_device = benchmark['feature_bank_device']
#                 benchmark['feature_bank'] = []
#                 benchmark['targets_bank'] = []
#                 dataloader = benchmark['dataloader']
#                 if callable(dataloader):
#                     dataloader = dataloader()
#                 for data in dataloader:
#                     img, target = data
#                     img = img.to(self.device)
#                     feature = backbone(img).squeeze()
#                     feature = F.normalize(feature, dim=1)
#                     feature = feature.to(feature_bank_device)
#                     target = target.to(feature_bank_device)
#                     benchmark['feature_bank'].append(feature)
#                     benchmark['targets_bank'].append(target)
#                 benchmark['feature_bank'] = torch.cat(
#                     benchmark['feature_bank'], dim=0).t().contiguous()
#                 benchmark['targets_bank'] = torch.cat(
#                     benchmark['targets_bank'], dim=0).t().contiguous()
#         self.train()

#     def training_epoch_end(self, outputs):
#         # update feature bank at the end of each training epoch
#         self.update_feature_bank()
#         if 'image' in self.class_embedding:
#             grid = torchvision.utils.make_grid(
#                 self.model['embedding'].get_images(), normalize=True, scale_each=True)
#             self.logger.experiment[0].add_image(
#                 'images', grid, self.global_step)

#     def validation_step(self, batch, batch_idx):
#         # we can only do kNN predictions once we have a feature bank
#         if not hasattr(self, 'benchmarks'):
#             return
#         images, targets = batch
#         model = getattr(self, '_model_ema', self.model)
#         model.eval()
#         feature = model['backbone'](images).squeeze()
#         feature = F.normalize(feature, dim=1)

#         acc = []
#         for name, benchmark in self.benchmarks.items():
#             if 'feature_bank' not in benchmark.keys():
#                 continue
#             feature_bank_device = benchmark['feature_bank_device']
#             feature = feature.to(feature_bank_device)
#             targets = targets.to(feature_bank_device)
#             pred_labels = knn_predict(
#                 feature,
#                 benchmark['feature_bank'],
#                 benchmark['targets_bank'],
#                 benchmark['num_classes'],
#                 benchmark['knn_k'],
#                 benchmark['knn_t']
#             )
#             top1 = (pred_labels[:, 0] == targets).float().mean().to(self.device)
#             if len(name) > 0:
#                 var_name = f'accuracy/val({name})'
#             else:
#                 var_name = 'accuracy/val'
#             self.log(var_name, top1, on_epoch=True, prog_bar=True)
#             acc.append(top1)

#         return acc


# @torch.jit.script
def uniform_label_loss(x: torch.Tensor):
    lsm = x.log_softmax(dim=-1)
    B, C = x.shape
    K = B // C
    loss = 0.
    for i in range(C):
        class_i_ll, _ = lsm[:, i].topk(K)
        loss += (-class_i_ll).sum()
    return loss / (K * C)


def assign_label(x, labels, T=1.0, alignment_matrix=None):
    C = x.size(-1)
    cand_labels = x.div(T).softmax(dim=-1)
    if alignment_matrix is None:
        alignment_matrix = torch.full((C, C), 0., device=x.device)
    alignment_matrix.index_add_(0, labels, cand_labels)
    return alignment_matrix


class UniformUnsupTask(ClassificationTask):
    def __init__(self, model, optimizer, **kwargs):
        self.alignment_matrix = None
        super().__init__(model, optimizer, **kwargs)

    def loss(self, output):
        # TODO: add mixup support
        # if self.mixup:
        # target = self.mixup.mix_target(target, output.size(-1))
        return uniform_label_loss(output)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):  # drop unlabled
            batch = batch['unlabeled']
        x, y = batch
        if self.mixup:
            self.mixup.sample(x.size(0))
            x = self.mixup(x)
        y_hat = self.model(x)
        loss = self.loss(y_hat)
        # acc = FM.accuracy(y_hat.softmax(-1), y)
        self.log_lr(on_step=True)
        self.log_dict(
            {
                'loss/train': loss
                # 'accuracy/train': acc\
            },
            prog_bar=True, on_epoch=True, on_step=True)
        if self.use_sam:
            eps_w = self.sam_step(loss)
            loss_w_sam = self.loss(self.model(x), y)
            # revert eps_w
            torch._foreach_sub_(list(self.parameters()), eps_w)
            self.manual_step(loss_w_sam)
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.alignment_matrix = None
        return super().training_epoch_end(outputs)

    def evaluation_step(self, batch, batch_idx):
        model = getattr(self, '_model_ema', self.model)
        model.eval()
        x, y = batch
        y_hat = model(x)
        loss = self.loss(y_hat)
        self.alignment_matrix = assign_label(y_hat, y,
                                             alignment_matrix=self.alignment_matrix)
        # acc = FM.accuracy(y_hat.softmax(dim=-1), y)
        return {
            # 'accuracy': acc,
            'loss': loss
        }

    def validation_epoch_end(self, outputs) -> None:
        grid = self.alignment_matrix.clone().unsqueeze(0)
        grid = grid / grid.max()
        for exp in self.logger.experiment:
            if isinstance(exp, torch.utils.tensorboard.writer.SummaryWriter):
                exp.add_image('class_alignment', grid, self.global_step)
        return super().validation_epoch_end(outputs)


if __name__ == '__main__':
    x = torch.randn(128, 10)

    loss = uniform_label_loss(x)
    print(loss)
    y = torch.tensor([[2., 3., 4., 5.]])
    labels = torch.tensor([2])
    out = assign_label(y, labels)
