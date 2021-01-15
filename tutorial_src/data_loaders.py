import csv
import os
from typing import Dict, List

import nltk
import torch
import torchtext
from torch.utils.data import Dataset

twitter_label = {'negative': 0, 'neutral': 1, 'positive': 2}


class TwitterDataset(Dataset):
    """https://www.kaggle.com/c/tweet-sentiment-extraction/data"""

    def __init__(self, split='train'):
        super().__init__()
        self.dataset = []
        if split == 'train':
            data_path = os.path.join('ALPS_2021/data/train.csv')
        elif split == 'val':
            data_path = os.path.join('ALPS_2021/data/test.csv')
        with open(data_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',',
                                    quoting=csv.QUOTE_MINIMAL)
            self.dataset.extend([line for line in csv_reader][1:])
        # 358bd9e861," Sons of ****, why couldn`t they put them on the
        # releases we already bought","Sons of ****,",negative
        for i in range(len(self.dataset)):
            self.dataset[i] = (self.dataset[i][1],
                               twitter_label[self.dataset[i][-1]])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


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

    return torch.nn.Parameter(word_vectors, requires_grad=True), word_to_index


class EmbeddingsVocabTokenizer:
    def __init__(self, word_to_id, id_to_word):
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.tokenizer = nltk.tokenize.word_tokenize
        self.pad_token_id = self.word_to_id['<pad>']
        self.mask_token_id = self.word_to_id['unk']

    def encode(self, text, max_length=-1, lower=True):
        tokens = self.tokenizer(text)
        if lower:
            tokens = [t.lower() for t in tokens]

        token_ids = [self.word_to_id[token] if token in self.word_to_id else
                     self.word_to_id['unk'] for token in tokens]
        return token_ids[:max_length]

    def __len__(self):
        return len(self.word_to_id)

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        return [self.id_to_word[token] for token in token_ids]


def collate_tweet(instances: List[Dict],
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
    output_tensors = list(_t.to(device) for _t in output_tensors)

    if return_seq_lens:
        seq_lengths = []
        for instance in output_tensors[0]:
            for _i in range(len(instance) - 1, -1, -1):
                if instance[_i] != tokenizer.pad_token_id:
                    seq_lengths.append(_i + 1)
                    break
        output_tensors.append(seq_lengths)

    return output_tensors
