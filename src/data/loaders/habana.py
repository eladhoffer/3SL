from habana_dataloader import HabanaDataLoader as _HabanaDataLoader


class HabanaDataLoader(_HabanaDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aeon_dl_handle_vars(kwargs)
        self.collate_fn = None
        self.worker_init_fn = None
