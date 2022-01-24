
class TokenizeDataset(object):
    def __init__(self, tokenizer, tokenizer_args, input='sentence', columns=['input_ids', 'attention_mask']):
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.columns = columns
        self.input = input

    def __call__(self, dataset):
        dataset = dataset.map(lambda e: self.tokenizer(e[self.input], **self.tokenizer_args),
                              batched=True)
        dataset.set_format(type='torch', columns=self.columns)
        return dataset


class TokenizeString(object):
    def __init__(self, tokenizer, tokenizer_args={}, add_bos=False, add_eos=False):
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.add_bos = add_bos
        self.add_eos = add_eos

    def __call__(self, text):
        if self.add_bos:
            text = self.tokenizer.bos_token + text
        if self.add_eos:
            text = text + self.tokenizer.eos_token
        return self.tokenizer(text, **self.tokenizer_args)
