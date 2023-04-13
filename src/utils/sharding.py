import torch.distributed.fsdp


def ShardingStrategy(name=None):
    return getattr(torch.distributed.fsdp.ShardingStrategy, name)
