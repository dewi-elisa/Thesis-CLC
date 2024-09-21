# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchrl
import torch.nn as nn
from transformers import BertTokenizer, BertModel


def get_bert_embedding(sentences, tokenizer, model):
    print(sentences)
    # Tokenize sentence
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Get last hidden state
    last_hidden_state = outputs.last_hidden_state

    # Get sentence embeddings
    cls_embedding = last_hidden_state[:, 0, :]

    return cls_embedding


class Sender(nn.Module):
    def __init__(self, n_features, n_hidden, images):
        super(Sender, self).__init__()
        if images:
            self.images = True
            self.num_cells = [10, 250, 100, 348, 200, 288, 150, 128]
            self.conv = torchrl.modules.ConvNet(in_features=3, depth=8, num_cells=self.num_cells)
            self.fc1 = nn.Linear(256*self.num_cells[-1], n_hidden)
            self.n_features = n_features
        else:
            self.images = False
            self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x, _aux_input=None):
        batch_size = x.shape[0]
        if self.images:
            x = x.reshape((-1, 3, 32, 32))
            x = self.conv(x)
            x = x.reshape((batch_size, -1))
        x = self.fc1(x).tanh()
        return x


class Receiver(nn.Module):
    def __init__(self, n_features, linear_units, images):
        super(Receiver, self).__init__()
        if images:
            self.images = True
            self.num_cells = [10, 250, 100, 348, 200, 288, 150, 128]
            self.conv = torchrl.modules.ConvNet(in_features=3, depth=8, num_cells=self.num_cells)
            self.fc1 = nn.Linear(256*self.num_cells[-1], linear_units)
        else:
            self.images = False
            self.fc1 = nn.Linear(n_features, linear_units)

    def forward(self, x, _input, _aux_input=None):
        batch_size = _input.shape[0]
        if self.images:
            _input = _input.reshape((-1, 3, 32, 32))
            _input = self.conv(_input)
            _input = _input.reshape((batch_size, 5, -1))
        embedded_input = self.fc1(_input).tanh()
        energies = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))

        return energies.squeeze()
