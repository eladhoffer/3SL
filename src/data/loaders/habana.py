from habana_dataloader import HabanaDataLoader as _HabanaDataLoader


class HabanaDataLoader(_HabanaDataLoader):
    def __init__(self, *args, **kwargs):
        args = [kwargs.pop('dataset')]
        super().__init__(*args, **kwargs)
        self.collate_fn = None
        self.worker_init_fn = None
