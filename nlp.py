from torchtext.datasets import IMDB
from torch.utils.data.dataset import random_split
import torch
import torch.nn as nn
import sys

import re
from torchtext.vocab import vocab
from pkg_resources import parse_version
from collections import Counter, OrderedDict
from torchtext import __version__ as torchtext_version
from torch.utils.data import DataLoader

############################################################


def tokenizer(text):
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text.lower())
    text = re.sub("[\W]+", " ", text.lower()) + " ".join(emoticons).replace("-", "")
    tokenized = text.split()
    return tokenized


def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))
    label_list = torch.tensor(label_list)
    lengths = torch.tensor(lengths)
    padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    return padded_text_list.to(device), label_list.to(device), lengths.to(device)


#############################################################################
train_dataset = IMDB(split="train")
test_dataset = IMDB(split="test")
torch.manual_seed(1)
train_dataset, valid_dataset = random_split(list(train_dataset), [20000, 5000])

token_counts = Counter()

for label, line in train_dataset:
    tokens = tokenizer(line)
    token_counts.update(tokens)

print("Vocab-size:", len(token_counts))

sorted_by_freq_tuples = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

ordered_dict = OrderedDict(sorted_by_freq_tuples)

vocab = vocab(ordered_dict)

vocab.insert_token("<pad>", 0)
vocab.insert_token("<unk>", 1)
vocab.set_default_index(1)

print([vocab[token] for token in ["this", "is", "an", "example"]])

## Step 3-A: define the functions for transformation

device = torch.device("cpu")

text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: 1.0 if x == 2 else 0.0
# 1 ~ negative, 2 ~ positive review

## Step 3-B: wrap the encode and transformation function
## Take a small batch


dataloader = DataLoader(
    train_dataset, batch_size=4, shuffle=False, collate_fn=collate_batch
)
text_batch, label_batch, length_batch = next(iter(dataloader))
print(text_batch)
print(label_batch)
print(length_batch)
print(text_batch.shape)


## Step 4: batching the datasets

batch_size = 32

train_dl = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
)
valid_dl = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
)
test_dl = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
)

## Embedding
embedding = nn.Embedding(num_embeddings=10, embedding_dim=3, padding_idx=0)

# a batch of 2 samples of 4 indices each

text_encoded_input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 0]])

print(embedding(text_encoded_input))
