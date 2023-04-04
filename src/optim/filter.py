from src.utils_pt.param_filter import FilterParameters, is_bn
from torch.nn import LayerNorm


def is_ln(module):
    return isinstance(module, LayerNorm)


def is_bias(name):
    return name.endswith('bias')


class Filter(FilterParameters):
    def __init__(self, model, module=None, module_name=None, parameter_name=None, exclude=False):
        super().__init__(model, module, module_name, parameter_name, exclude)


class OnlyBN(Filter):
    def __init__(self, model, *kargs, **kwargs):
        super().__init__(model, module=is_bn, *kargs, **kwargs)


class OnlyBiasBN(Filter):
    def __init__(self, model, *kargs, **kwargs):
        super().__init__(model, module=is_bn, parameter_name=is_bias, *kargs, **kwargs)


class OnlyBiasLN(Filter):
    def __init__(self, model, *kargs, **kwargs):
        super().__init__(model, module=is_ln, parameter_name=is_bias, *kargs, **kwargs)
