import torch
import numpy as np
import os
from torch.utils.data import Dataset


class TextBlockDataset(Dataset):
    def __init__(self, data_dir, split='train', seq_length=1024, random_position=False):
        self.data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
        self.seq_length = seq_length
        self.random_position = random_position

    def __getitem__(self, idx):
        block = self.seq_length + 1
        if self.random_position:
            offset = np.random.randint(0, len(self.data) - block)
        else:
            offset = idx * block
        x = torch.from_numpy(self.data[offset: offset + block].astype(np.int64))
        return x[:-1], x[1:]

    def __len__(self):
        return len(self.data) // (self.seq_length + 1)


# class HFTextDataset(torch.utils.data.Dataset):
#     def __init__(self, tokenizer, name="openwebtext", split="train", cache_dir=None, tokenizer_kwargs={}, dataset_kwargs={}):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.split = split
#         self.cache_dir = cache_dir
#         if cache_dir is not None:
#             cache_file = f"{cache_dir}/{name}_tok_{tokenizer.name_or_path}.arrow"
#         else:
#             cache_file = None
#         tokenizer_args = {"return_token_type_ids": False,
#                           "truncation": True,
#                           "max_length": max_length}
#         preproc = PreprocessWikiBooks(tokenizer, break_mode=break_mode,
#                                       tokenizer_args=tokenizer_args,
#                                       block_size=max_length)

#         wiki = load_dataset("wikipedia", wikipedia_name, split=split)
#         wiki = wiki.remove_columns("title")  # only keep the text
#         wiki = wiki.map(preproc.map_fn, batched=True, num_proc=32,
#                         remove_columns=wiki.column_names,
#                         cache_file_name=cache_file)
#         bookcorpus = load_dataset("bookcorpus", split=split)
#         bookcorpus = bookcorpus.map(preproc.join_samples, batched=True, batch_size=None,
#                                     remove_columns=bookcorpus.column_names,
#                                     cache_file_name=bookcorpus_cache_file)
#         self.dataset = concatenate_datasets([wiki, bookcorpus])
#         self.dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

#     def __getitem__(self, index):
#         return self.dataset[index]

#     def __len__(self):
#         return len(self.dataset)


# if __name__ == '__main__':
#     dset = TextBlockDataset('/home/ehoffer/Pytorch/nanoGPT/data/openwebtext', 'train', 1024)


# if __name__ == "__main__":
#     from transformers import AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     train_dset = Text(tokenizer, split="train", cache_dir="/labdata2/Datasets/wikibooks")
#     split_dataset = train_dset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
#     # val_dset = WikiBooks(tokenizer, split="train[99.95%:]",
#     #  cache_dir="/home/ehoffer/Datasets/wikibooks")
