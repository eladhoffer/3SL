
class Tokenize(object):
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