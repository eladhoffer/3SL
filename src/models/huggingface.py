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
    def __init__(self, image_model, text_model, image_feat_size=2048, text_feat_size=512,
                 freeze_image_model=False, freeze_text_model=True, remove_pooler=False,
                 prompt_embedding=False):
        super().__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.freeze_image_model = freeze_image_model
        self.freeze_text_model = freeze_text_model
        self.image_feat_size = image_feat_size
        self.prompt_embedding = prompt_embedding
        self.adapter = torch.nn.Sequential(
            torch.nn.LayerNorm(image_feat_size),
            torch.nn.Linear(image_feat_size, text_feat_size),
            torch.nn.LayerNorm(text_feat_size)
        )
        if self.prompt_embedding:
            self.prompt = torch.nn.Embedding(1, text_feat_size)
        if remove_pooler:
            self.image_model.avgpool = torch.nn.Identity()
        self.image_model.fc = torch.nn.Identity()
        if self.text_model.config.is_encoder_decoder:
            self.text_model.encoder = None
        if freeze_image_model:
            for param in self.image_model.parameters():
                param.requires_grad = False
        if freeze_text_model:
            for param in self.text_model.parameters():
                param.requires_grad = False

    def _vision_encoder(self, image):
        self.image_model.train(not self.freeze_image_model)
        features = self.image_model(image)
        features = features.view(features.size(0), self.image_feat_size, -1)
        features = features.permute(0, 2, 1)
        if self.freeze_image_model:
            features.requires_grad = False
        features = self.adapter(features)
        if self.prompt_embedding:
            prompt = self.prompt.weight.view(1, 1, -1).expand(features.size(0), -1, -1)
            features = torch.cat((prompt, features), dim=1)
        return features

    def forward(self, image, **kwargs):
        self.text_model.train(not self.freeze_text_model)
        features = self._vision_encoder(image)
        if self.text_model.config.is_encoder_decoder:
            kwargs.pop('input_ids', None)
            output = self.text_model(encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=features), **kwargs)
        else:
            input_ids = kwargs.pop('input_ids')
            input_ids = input_ids[:, :-1]
            inputs_embeds = self.text_model.transformer.wte(input_ids)
            inputs_embeds = torch.cat((features, inputs_embeds), dim=1)
            output = self.text_model(inputs_embeds=inputs_embeds, **kwargs)
        return output

    def generate(self, image, **kwargs):
        encoder_outputs = self._vision_encoder(image)
        output = self.text_model.generate(encoder_outputs=encoder_outputs, **kwargs)
        return output
