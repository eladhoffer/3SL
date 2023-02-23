from hydra.utils import instantiate
from src.utils_pt.optim import OptimRegime
from torch.optim.optimizer import Optimizer

# create Optimizer wrapper with all functions

class OptimizerWrapper(Optimizer):
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)

    def step(self):
        self.optimizer.step()

    def state_dict(self) -> dict:
        return self.optimizers.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        return self.optimizers.load_state_dict(state_dict)


class OptimConfig(object):
    """
    Optimization configuration.
    """

    def __init__(self, optimizer, lr_scheduler=None, regularizer=None, filter=None,
                 interval='step', frequency=1, monitor=None, strict=True, name=None,
                 optimizer_frequency=None):
        """
        :param optim_regime: Optimization regime.
        """
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.regularizer = regularizer
        self.filter = filter
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor
        self.strict = strict
        self.name = name
        self.optimizer_frequency = optimizer_frequency

    @staticmethod
    def instantiate(model, optimizer, lr_scheduler=None, regularizer=None,
                    filter=None, optimizer_filters=None, interval='step', frequency=1,
                    monitor=None, strict=True, name=None):
        if filter is not None:
            model = instantiate(filter, model=model, _convert_="all")
        if optimizer_filters is None:
            params = model.parameters()
        else:
            params = []
            for filter_config in optimizer_filters:
                filter_config['params'] = instantiate(filter_config.pop('filter'),
                                                      model=model, _convert_="all").parameters()
                params.append(filter_config)
        optimizer = instantiate(optimizer, params=params, _convert_="all")
        if regularizer is not None:
            regularizer = instantiate(regularizer, model=model, _convert_="all")

        if lr_scheduler is not None:
            lr_scheduler = instantiate(lr_scheduler,
                                       optimizer=optimizer, _convert_="all")
        return OptimConfig(optimizer, lr_scheduler, regularizer, filter,
                           interval=interval, frequency=frequency,
                           monitor=monitor, strict=strict, name=name)

    def lr_dict(self):
        return {
            # REQUIRED: The scheduler instance
            'scheduler': self.lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            'interval': self.interval,
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            'frequency': self.frequency,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            'monitor': self.monitor,
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            'strict': self.strict,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            'name': self.name,
        }

    def configuration(self):
        config = {"optimizer": self.optimizer}
        if self.lr_scheduler is not None:
            config["lr_scheduler"] = self.lr_dict()
        if self.optimizer_frequency is not None:
            config["frequency"] = self.optimizer_frequency
        return config

    def regularizers(self):
        return [self.regularizer]


class OptimConfigList(OptimConfig):
    def __init__(self, configs):
        self.optim_configs = configs

    @staticmethod
    def instantiate(model, configs):
        optim_configs = [OptimConfig.instantiate(model, **config) for config in configs]
        return OptimConfigList(optim_configs)

    def configuration(self):
        if self.optim_configs[0].optimizer_frequency is not None:
            return tuple((optim_config.configuration() for optim_config in self.optim_configs))
        else:
            optimizer_list = [optim_config.optimizer for optim_config in self.optim_configs]
            lr_scheduler_list = [optim_config.lr_dict() for optim_config in self.optim_configs]
            return optimizer_list, lr_scheduler_list

    def regularizers(self):
        return tuple([optim_config.regularizer for optim_config in self.optim_configs])
