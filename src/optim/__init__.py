from hydra.utils import instantiate
from src.utils_pt.optim import OptimRegime


class OptimConfig(object):
    """
    Optimization configuration.
    """

    def __init__(self, optimizer, lr_scheduler=None, regularizer=None, filter=None,
                 interval='step', frequency=1, monitor=None, strict=True, name=None):
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

    @staticmethod
    def instantiate(model, optimizer, lr_scheduler=None, regularizer=None, filter=None,
                    interval='step', frequency=1, monitor=None, strict=True, name=None):
        if filter is not None:
            model = instantiate(filter, model=model, _convert_="all")
        optimizer = instantiate(optimizer, params=model.parameters(),
                                _convert_="all")
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
        return {"optimizer": self.optimizer,
                "lr_scheduler": self.lr_dict()}

    def regularizers(self):
        return [self.regularizer]


class OptimConfigList(OptimConfig):
    def __init__(self, *optim_configs):
        self.optim_configs = optim_configs

    @staticmethod
    def instantiate(model, *optim_configs):
        optim_configs = [OptimConfig.instantiate(model, *config) for config in optim_configs]
        return OptimConfigList(optim_configs)

    def configuration(self):
        return (optim_config.configuration() for optim_config in self.optim_configs)

    def regularizers(self):
        return [optim_config.regularizer for optim_config in self.optim_configs]
