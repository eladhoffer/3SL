from src.utils_pt.param_filter import FilterParameters, is_bn, is_not_bn


class Filter(FilterParameters):
    def __init__(self, model, module=None, module_name=None, parameter_name=None):
        super().__init__(model, module, module_name, parameter_name)


class OnlyBN(Filter):
    def __init__(self, model):
        super().__init__(model, module=is_bn)


class ExcludeBN(Filter):
    def __init__(self, model):
        super().__init__(model, module=is_not_bn)
