from torch.utils.data._utils.collate import default_collate


def collate_by_name(**collator_fns):
    def _collator_fn(batches):
        batch_by_key = {}
        for item in batches:
            for key, value in item.items():
                batch_by_key.setdefault(key, []).append(value)
        for key, batch in batch_by_key.items():
            collate_fn = collator_fns.get(key, default_collate)
            batch_by_key[key] = collate_fn(batch)
        return batch_by_key
    return _collator_fn
