# adapted from https://github.com/NVIDIA/DALI/blob/main/docs/examples/use_cases/pytorch/resnet50/main.py

from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from torch import distributed as dist


@pipeline_def
def imagenet_pipeline(data_dir, crop, size, shard_id, num_shards, cpu=False, is_training=True,
                      random_aspect_ratio=[0.8, 1.25], random_area=[0.1, 1.0],
                      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                      image_dtype='float', label_dtype='long'):
    image_dtype = types.to_dali_type(image_dtype)
    label_dtype = types.to_dali_type(label_dtype)
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")
    dali_device = 'cpu' if cpu else 'gpu'
    decoder_device = 'cpu' if cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=random_aspect_ratio,
                                               random_area=random_area,
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images,
                                      dtype=image_dtype,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[v * 255.0 for v in mean],
                                      std=[v * 255.0 for v in std],
                                      mirror=mirror)
    if not cpu:
        images = images.gpu()
        labels = labels.gpu()
    labels = fn.cast(labels, dtype=label_dtype)
    return images, labels


class ModifiedDALIClassificationIterator(DALIClassificationIterator):
    def __next__(self):
        batch = super(ModifiedDALIClassificationIterator, self).__next__()
        return batch[0]['data'], batch[0]['label'][:, 0]

    def _end_iteration(self):
        if self._auto_reset:
            self.reset()


def imagenet_loader(dataset, batch_size, drop_last=False, num_workers=0,
                    input_size=224, scale_size=256,
                    random_aspect_ratio=[0.8, 1.25], random_area=[0.1, 1.0],
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                    image_dtype='float', label_dtype='long',
                    is_training=True, cpu=True, group=None):
    if dist.is_initialized():
        local_rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)
    else:
        local_rank = 0
        world_size = 1
    pipe = imagenet_pipeline(batch_size=batch_size,
                             num_threads=num_workers,
                             device_id=local_rank,
                             seed=12 + local_rank,
                             data_dir=dataset,
                             crop=input_size,
                             size=scale_size,
                             cpu=cpu,
                             shard_id=local_rank,
                             num_shards=world_size,
                             is_training=is_training,
                             mean=mean,
                             std=std,
                             random_aspect_ratio=random_aspect_ratio,
                             random_area=random_area,
                             image_dtype=image_dtype,
                             label_dtype=label_dtype)
    pipe.build()
    return ModifiedDALIClassificationIterator(pipe,
                                              reader_name="Reader", auto_reset=True,
                                              last_batch_policy=LastBatchPolicy.DROP if drop_last
                                              else LastBatchPolicy.PARTIAL)
