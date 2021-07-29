import transformers
from transformers.models.roberta.modeling_roberta import RobertaLMHead


class OptimizerRobertaLMHead(RobertaLMHead):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__(config)
        self.masked_tokens = None

    def set_masked_tokens(self, masked_tokens):
        self.masked_tokens = masked_tokens

    def forward(self, features, **kwargs):
        if self.masked_tokens is not None:
            features = features[self.masked_tokens, :]
        return super().forward(features, **kwargs)


class RobertaForMaskedLM(transformers.RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.lm_head = OptimizerRobertaLMHead(config)
        if config.tie_word_embeddings:
            # share word embedding with classifier weights
            self.lm_head.decoder.weight = self.roberta.embeddings.word_embeddings.weight
