# IPython log file

from transformers import pipeline, AutoTokenizer
import torch
from transformers import RobertaConfig, RobertaForMaskedLM

c = torch.load('./results/roberta_base2/checkpoints/last.ckpt', map_location='cpu')
model = RobertaForMaskedLM(RobertaConfig(vocab_size=50265))
state_dict = {name.replace('model.', '').replace('lm_head.lm_head', 'lm_head'): weight
              for name, weight in c['state_dict'].items()}
model.load_state_dict(state_dict)

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
task = pipeline('fill-mask', model=model, tokenizer=tokenizer)
print(task("hello <mask>"))
print(task("what is <mask> with you"))
print(task("Barack <mask> was the president of United States"))
