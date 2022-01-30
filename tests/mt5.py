import torch
from transformers import T5ForConditionalGeneration, MT5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
# model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
# tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
# article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
# summary = "Weiter Verhandlung in Syrien."
# inputs = tokenizer(article, return_tensors="pt")
# with tokenizer.as_target_tokenizer():
#     labels = tokenizer(summary, return_tensors="pt")

# outputs = model(**inputs, labels=labels["input_ids"])
# loss = outputs.loss


model = T5ForConditionalGeneration.from_pretrained("t5-small")
encoder = model.encoder
model.encoder = None
tokenizer = T5Tokenizer.from_pretrained("t5-small")
# inference

inputs = tokenizer(
    "translate: This is a lovely day", return_tensors="pt")
decoder_inputs = tokenizer("<sos> Weiter Verhandlung in Syrien.", return_tensors="pt")
labels = tokenizer("Weiter Verhandlung in Syrien. <pad>", return_tensors="pt")
last_hidden_state = encoder(**inputs).last_hidden_state
encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=last_hidden_state)
# outputs = model(encoder_outputs=encoder_outputs, decoder_input_ids=decoder_inputs["input_ids"])#, labels=decoder_inputs["input_ids"])
# outputs = model.generate(input_ids=inputs.input_ids)
outputs = model.generate(encoder_outputs=encoder_outputs, num_beams=5)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# loss1 = outputs.loss
# loss2 = torch.nn.functional.cross_entropy(outputs.logits.flatten(0,1), labels.flatten(0,1))