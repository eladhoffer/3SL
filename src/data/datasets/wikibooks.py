import torch
from datasets import concatenate_datasets, load_dataset


class PreprocessWikiBooks(object):
    def __init__(self, tokenizer, tokenizer_args={}, field="text",
                 break_mode="complete_doc", remove_sep=False, remove_cls=False, block_size=512) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.break_mode = break_mode
        self.remove_sep = remove_sep
        self.block_size = block_size
        self.field = field
        self.remove_cls = remove_cls

    def _cat(self, x, y):
        if self.remove_sep:
            if torch.is_tensor(x):
                x = x[:, :-1]
            else:
                x = x[:-1]
        if self.remove_cls:
            if torch.is_tensor(y):
                y = y[:, 1:]
            else:
                y = y[1:]
        if torch.is_tensor(x):
            return torch.cat([x, y], dim=-1)
        else:
            return x + y

    def join_samples(self, sample):
        output = {}
        for text in sample[self.field]:
            for key, val in self.tokenizer(text, **self.tokenizer_args).items():
                if key not in output:
                    output[key] = [val]
                else:
                    if len(output[key][-1]) + len(val) >= self.block_size:
                        output[key].append(val)
                    else:
                        output[key][-1] = self._cat(output[key][-1], val)
        return output

    def break_complete_doc(self, sample):
        output = {}
        for text in sample[self.field]:
            new_doc = True
            for line in text.splitlines():
                if len(line) == 0:
                    continue
                for key, val in self.tokenizer(line, **self.tokenizer_args).items():
                    if key not in output:
                        output[key] = [val]
                    else:
                        if new_doc or \
                                len(output[key][-1]) + len(val) >= self.block_size:
                            output[key].append(val)
                        else:
                            output[key][-1] = self._cat(output[key][-1], val)
                new_doc = False
        return output

    def map_fn(self, sample):
        if self.break_mode == "complete_doc":
            sample = self.break_complete_doc(sample)
        else:
            raise ValueError("Unknown break_mode: {}".format(self.break_mode))
        return sample


class WikiBooks(torch.utils.data.Dataset):
    def __init__(self, tokenizer, wikipedia_name="20200501.en", split="train",
                 max_length=128, break_mode="complete_doc", cache_dir=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.wikipedia_name = wikipedia_name
        self.split = split
        self.max_length = max_length
        self.break_mode = break_mode
        self.cache_dir = cache_dir
        if cache_dir is not None:
            wiki_cache_file = f"{cache_dir}/wikipedia_split_{split}_length_{max_length}_tok_{tokenizer.name_or_path}.arrow"
            bookcorpus_cache_file = f"{cache_dir}/bookcorpus_split_{split}_length_{max_length}_tok_{tokenizer.name_or_path}.arrow"
        else:
            wiki_cache_file = None
            bookcorpus_cache_file = None
        tokenizer_args = {"return_token_type_ids": False,
                          "truncation": True,
                          "max_length": max_length}
        preproc = PreprocessWikiBooks(tokenizer, break_mode=break_mode,
                                      tokenizer_args=tokenizer_args,
                                      block_size=max_length)

        wiki = load_dataset("wikipedia", wikipedia_name, split=split)
        wiki = wiki.remove_columns("title")  # only keep the text
        wiki = wiki.map(preproc.map_fn, batched=True, num_proc=32,
                        remove_columns=wiki.column_names,
                        cache_file_name=wiki_cache_file)
        bookcorpus = load_dataset("bookcorpus", split=split)
        bookcorpus = bookcorpus.map(preproc.join_samples, batched=True, batch_size=None,
                                    remove_columns=bookcorpus.column_names,
                                    cache_file_name=bookcorpus_cache_file)
        self.dataset = concatenate_datasets([wiki, bookcorpus])
        self.dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    train_dset = WikiBooks(
        tokenizer, split="train[:99%]", cache_dir="/home/ehoffer/Datasets/wikibooks")
    val_dset = WikiBooks(tokenizer, split="train[99%:]",
                         cache_dir="/home/ehoffer/Datasets/wikibooks")
