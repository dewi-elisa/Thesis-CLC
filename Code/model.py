import torch
import torch.nn as nn
import egg.core as core

from torchvision import datasets, transforms
from torch import nn
from torch.nn import functional as F

import matplotlib.pyplot as plt
import random
import numpy as np

from transformers import BertTokenizer, BertModel

opts = core.init(params=['--random_seed=7',  # will initialize numpy, torch, and python RNGs
                         '--lr=1e-3',   # sets the learning rate for the selected optimizer
                         '--batch_size=32',
                         '--optimizer=adam'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_bert_embedding(sentence):
    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors='pt')

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # The last hidden state is the first element of the outputs tuple
    last_hidden_state = outputs.last_hidden_state

    # Get the embeddings for the [CLS] token (the first token)
    cls_embedding = last_hidden_state[:, 0, :].squeeze()

    return cls_embedding

# Example usage
sentence = "I want to generate an embedding for this sentence."
embedding = get_bert_embedding(sentence)
print(embedding.size())

embedding = get_bert_embedding


class Sender(nn.Module):
    def __init__(self, embedding, output_size):
        super(Sender, self).__init__()
        self.fc = nn.Linear(500, output_size)
        self.embedding = embedding

    def forward(self, x, aux_input=None):
        with torch.no_grad():
            x = self.vision(x)
        x = self.fc(x)
        return x


class Receiver(nn.Module):
    def __init__(self, input_size):
        super(Receiver, self).__init__()
        self.fc = nn.Linear(input_size, 784)

    def forward(self, channel_input, receiver_input=None, aux_input=None):
        x = self.fc(channel_input)
        return torch.sigmoid(x)


def loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input=None):
    loss = F.binary_cross_entropy(receiver_output, sender_input.view(-1, 784), reduction='none').mean(dim=1)
    return loss, {}


sender = Sender(embedding, output_size=400)
receiver = Receiver(input_size=400)

vocab_size = 10

sender = Sender(embedding, output_size=vocab_size)
sender = core.ReinforceWrapper(sender)  # wrapping into a Reinforce interface

receiver = Receiver(input_size=400)
receiver = core.SymbolReceiverWrapper(receiver, vocab_size, agent_input_size=400)
receiver = core.ReinforceDeterministicWrapper(receiver)

game = core.SymbolGameReinforce(sender, receiver, loss, sender_entropy_coeff=0.05, receiver_entropy_coeff=0.0)
optimizer = torch.optim.Adam(game.parameters(), lr=1e-2) #  we can also use a manually set optimizer

trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                           validation_data=test_loader)

n_epochs = 15
trainer.train(n_epochs)

game.eval()

for z in range(vocab_size):
    t = torch.zeros(vocab_size).to(device)
    t[z] = 1
    with torch.no_grad():
        sample, _1, _2 = game.receiver(t)
        sample = sample.float().cpu()
    sample = sample.view(28, 28)
    plt.title(f"Input: symbol {z}")
    plt.imshow(sample, cmap='gray')
    plt.show()

plot(game, test_dataset, is_gs=False, variable_length=False)


