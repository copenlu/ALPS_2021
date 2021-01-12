from torch.utils.data import Dataset
import os
import torchtext
import torch
from typing import List, Dict
import nltk
import urllib.request
import subprocess
from urllib.parse import urlparse


def download_data(url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'):
  filename = os.path.basename(urlparse(url).path)
  urllib.request.urlretrieve(url, filename=filename)
  subprocess.call(['tar', 'xvzf', filename])


class IMDBDataset(Dataset):
    def __init__(self, split='train', data_dir='aclImdb'):
        if split == 'test':
            filenames_neg = os.scandir(os.path.join(data_dir, 'test', 'neg'))
            filenames_pos = os.scandir(os.path.join(data_dir, 'test', 'pos'))
        elif split == 'val':
            filenames_neg = os.scandir(os.path.join(data_dir, 'train', 'neg'))
            filenames_neg = [f for f in filenames_neg if f.name.startswith('8')]
            filenames_pos = os.scandir(os.path.join(data_dir, 'train', 'pos'))
            filenames_pos = [f for f in filenames_pos if f.name.startswith('8')]
        elif split == 'train':
            filenames_neg = os.scandir(os.path.join(data_dir, 'train', 'neg'))
            filenames_neg = [f for f in filenames_neg if
                             not f.name.startswith('8')]
            filenames_pos = os.scandir(os.path.join(data_dir, 'train', 'pos'))
            filenames_pos = [f for f in filenames_pos if
                             not f.name.startswith('8')]
        else:
            raise ValueError(f'Unknown split {split} value!')

        self.dataset = []
        self.dataset += [(open(fn).read(), 0) for fn in filenames_neg]
        self.dataset += [(open(fn).read(), 1) for fn in filenames_pos]

    def __getitem__(self, item: int):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


def get_embeddings(embeddings_name: str,
                   embedding_dim: int,
                   special_tokens: List[str] = ['<pad>']):
    """
    :return: a tensor with the embedding matrix - ids of words are from vocab
    """
    if embeddings_name == 'glove':
        embeddings = torchtext.vocab.GloVe(dim=embedding_dim, name='6B')

    word_to_index = embeddings.stoi
    word_vectors = embeddings.vectors

    for token in special_tokens:
        word_to_index[token] = len(word_to_index)
        word_vectors = torch.cat([word_vectors, torch.zeros(1, embedding_dim)],
                                 dim=0)

    # TODO: extract this, too
    return torch.nn.Parameter(word_vectors, requires_grad=True), word_to_index


class EmbeddingsTokenizer:
    def __init__(self, word_to_id, id_to_word):
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.tokenizer = nltk.tokenize.word_tokenize
        self.pad_token_id = self.word_to_id['<pad>']

    def encode(self, text, max_length=-1, lower=True):
        if lower:
            text = text.lower()
        tokens = self.tokenizer(text)
        token_ids = [self.word_to_id[token] if token in self.word_to_id else
                     self.word_to_id['unk'] for token in tokens]
        return token_ids[:max_length]

    def __len__(self):
        return len(self.word_to_id)


def collate_imdb(instances: List[Dict],
                 tokenizer,
                 return_attention_masks: bool = True,
                 pad_to_max_length: bool = False,
                 max_seq_len: int = 512,
                 device='cuda',
                 return_seq_lens: bool = False) -> List[torch.Tensor]:

    token_ids = [tokenizer.encode(_x[0], max_length=509) for _x in instances]
    if pad_to_max_length:
        batch_max_len = max_seq_len
    else:
        batch_max_len = max([len(_s) for _s in token_ids])
    padded_ids_tensor = torch.tensor(
        [_s + [tokenizer.pad_token_id] * (batch_max_len - len(_s)) for _s in
         token_ids])
    labels = torch.tensor([_x[1] for _x in instances], dtype=torch.long)

    output_tensors = [padded_ids_tensor]
    if return_attention_masks:
        output_tensors.append(padded_ids_tensor > 0)
    output_tensors.append(labels)

    if return_seq_lens:
        seq_lengths = []
        for instance in output_tensors[0]:
            for _i in range(len(instance) - 1, -1, -1):
                if instance[_i] != 0:
                    seq_lengths.append(_i + 1)
                    break
        output_tensors.append(seq_lengths)

    return list(_t.to(device) for _t in output_tensors)