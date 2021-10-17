from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os
import pandas as pd

# Mean Pooling - Take attention mask into account for correct averaging


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']


# Normalize embeddings
# sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)


def generate_cc12_sentence_embedding(filename, path, model_name='sentence-transformers/all-mpnet-base-v2',
                                     chunksize=100000, batch_size=512, device='cpu', dtype=torch.float32):
    if not os.path.exists(path):
        os.makedirs(path)
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device=device, dtype=dtype)
    data_iter = pd.read_table(filename, names=['url', 'caption'],
                              index_col=False, chunksize=chunksize)
    for data in data_iter:
        captions = data['caption'].tolist()
        start = data.index.start
        for idx in range(0, len(data), batch_size):
            sentences = captions[idx:idx + batch_size]
            print(f'Extracting sentences {idx+start}-{idx+start+len(sentences)}')
            # Tokenize sentences
            encoded_input = tokenizer(sentences, padding=True,
                                      truncation=True, return_tensors='pt')
            encoded_input = encoded_input.to(device=device)

            # Compute token embeddings
            with torch.no_grad():
                # encoded_input = encoded_input.to(device=device, dtype=dtype)
                model_output = model(**encoded_input)
                # Perform pooling
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            for j in range(len(sentences)):
                tensor_filename = os.path.join(path, f'{start+idx+j:08d}.pt')
                torch.save(sentence_embeddings[j].cpu(), tensor_filename)


def generate_sentence_embedding(sentences, model_name='sentence-transformers/all-mpnet-base-v2',
                                device='cpu', dtype=torch.float32):
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device=device, dtype=dtype)
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True,
                              truncation=True, return_tensors='pt')
    encoded_input = encoded_input.to(device=device)

    # Compute token embeddings
    with torch.no_grad():
        # encoded_input = encoded_input.to(device=device, dtype=dtype)
        model_output = model(**encoded_input)
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings


# download_images('./Validation_GCC-1.1.0-Validation.tsv', './validation')
# write_captions('./Validation_GCC-1.1.0-Validation.tsv', 'validation.txt')
# write_captions('./Train_GCC-training.tsv', 'training.txt')
# generate_cc12_sentence_embedding('/home/labuser/Datasets/cc12m/cc12m.tsv',
#                                  path='/home/labuser/Datasets/cc12m/sentence_embedding_mpnet_base',
#                                  model_name='sentence-transformers/all-mpnet-base-v2',
#                                  device='cuda:2', dtype=torch.half)


cifar10_classes = ['airplane', 'automobile', 'bird',
                   'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
cifar10_embedding = generate_sentence_embedding(cifar10_classes,
                                                model_name='sentence-transformers/all-mpnet-base-v2',
                                                device='cuda:1', dtype=torch.half)
print(cifar10_embedding)
torch.save(cifar10_embedding.cpu(), 'cifar10_embedding.pt')


# def concat_all_embeddings(embeddings_path, num=12423374, output='all_embeddings.pt'):
#     embeddings = torch.zeros((num, 768), dtype=torch.float, device='cpu')
#     for i in range(num):
#         filename = os.path.join(embeddings_path, f'{i:08d}.pt')
#         embeddings[i].copy_(torch.load(filename))
#     torch.save(embeddings, output)


# concat_all_embeddings('/home/labuser/Datasets/cc12m/sentence_embedding_mpnet_base',
#                       output='/home/labuser/Datasets/cc12m/sentence_embedding_mpnet_base/all_embeddings.pt')
