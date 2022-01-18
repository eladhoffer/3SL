import torch
import transformers
from transformers.models.roberta.modeling_roberta import RobertaLMHead
from transformers import T5ForConditionalGeneration, MT5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


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


class ImageToTextModel(torch.nn.Module):
    def __init__(self, image_model, text_model,
                 freeze_image_model=False, freeze_text_model=True,
                 text_model_type="encoder-decoder"):
        super().__init__()
        self.image_model = image_model
        self.text_model = text_model
        assert text_model_type in ["encoder-decoder"],\
            "text_model is assumed to be an encoder-decoder model"
        if text_model_type == "encoder-decoder":
            self.text_model.encoder = None
        if freeze_image_model:
            for param in self.image_model.parameters():
                param.requires_grad = False
        if freeze_text_model:
            for param in self.text_model.parameters():
                param.requires_grad = False

    def forward(self, image, text):
        features = self.image_model(image)
        features = features.view(features.size(0), -1, features.size(-1))
        encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=features)
        output = self.text_model(encoder_outputs=encoder_outputs,
                                 labels=text["input_ids"])
        return output
