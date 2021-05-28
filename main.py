import os
import sys
import time

import torch
import torch.nn as nn
import numpy as np
import random
import time

from preprocessing import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print(device)

class Model(nn.Module):
    def __init__(self, vocab_size, gru_size=512, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.gru_size = gru_size
        self.gru_layers = num_layers
        self.embedding = nn.Embedding(self.vocab_size, self.gru_size, padding_idx=0)
        self.gru = nn.GRU(
            input_size=self.gru_size,
            hidden_size=self.gru_size,
            num_layers=self.gru_layers,
        )
        self.linear = nn.Linear(in_features=self.gru_size, out_features=self.vocab_size)

    def forward(self, x, hidden_state):
        x = self.embedding(x)
        y, hidden_state = self.gru(x, hidden_state)
        y = self.linear(y)
        return y, hidden_state


def train(x, y, batch_size, num_epochs, vocab_size, window, gru_size=512, num_layers=1):
    num_batches = len(x//batch_size)
    hidden_state = torch.zeros(num_layers, batch_size, gru_size).clone().detach()
    model = Model(vocab_size, gru_size, num_layers)
    model.to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    steps_between_state_reset = 100
    start_time = time.time()
    min_loss = None
    for epoch in range(num_epochs):
        steps_since_state_reset = 0
        for batch in range(num_batches):
            if (batch+1)*batch_size > len(x):
                continue
            x_batch = torch.tensor(x[batch*batch_size:(batch+1)*batch_size])
            y_true = torch.tensor(y[batch*batch_size:(batch+1)*batch_size])
            x_batch = torch.transpose(x_batch, 0, 1)
            y_true = torch.transpose(y_true, 0, 1)
            y_pred, hidden_state = model(x_batch.to(device), hidden_state.to(device))
            y_pred = y_pred.reshape((window * batch_size, -1))
            y_true = y_true.reshape((window * batch_size))
            y_true = y_true.type(torch.LongTensor)
            optim.zero_grad()
            loss = nn.functional.cross_entropy(y_pred, y_true.to(device))
            loss.backward()
            optim.step()
            if min_loss is None or loss.item() < min_loss:
                min_loss = loss.item()
            # print("Epoch: {} batch: {} loss: {}".format(epoch, batch, loss.item()))
            steps_since_state_reset += 1
            if steps_since_state_reset >= steps_between_state_reset:
                steps_since_state_reset = 0
                hidden_state = torch.zeros(num_layers, batch_size, gru_size).clone().detach()
            else:
                hidden_state = hidden_state.clone().detach()
        print("Epoch: {}, min loss: {}".format(epoch, min_loss))
        torch.save(model.state_dict(), 'model_weights.pt')
    print("training time:", time.time() - start_time)


def generate(x, char_to_idx, idx_to_char, batch_size, num_epochs, vocab_size, window, gru_size=512, num_layers=1):
    model = Model(vocab_size, gru_size, num_layers)
    model.load_state_dict(torch.load('model_weights.pt'))
    model.to(device)
    model.eval()
    seq = random.choice(x)
    fin_seq = [idx_to_char[idx] for idx in seq] + ['|']
    seq = torch.tensor(seq)
    num_chars = 2000
    hidden_state = torch.zeros(num_layers, 1, gru_size).clone().detach()
    for i in range(num_chars):
        with torch.no_grad():
            seq = seq.reshape((window, 1))
            y_pred, _ = model(seq.to(device), hidden_state.to(device))
            y_pred = y_pred[-1][0]
            char_probs = nn.functional.softmax(y_pred, dim=0)
            char_idx = np.random.choice(vocab_size, p=char_probs.cpu().numpy())
            char_idx = torch.tensor(char_idx)
            seq = seq.squeeze().numpy()
            seq = seq[1:]
            seq = np.append(seq, char_idx.cpu())
            seq = torch.tensor(seq)
            fin_seq.append(idx_to_char[char_idx.item()])
    print("".join(fin_seq))




path = "pre_meduza_signs.txt"
window = 200
gru_size = 512
num_layers = 3
num_epochs = 50
batch_size = 64
text = Preprocessing.read_dataset(path)
char_to_idx, idx_to_char = Preprocessing.create_dictionary(text)
vocab_size = len(char_to_idx)
x, y = Preprocessing.build_sequences_target(text, char_to_idx, window)
train(x,y, batch_size, num_epochs, vocab_size, window, gru_size, num_layers)
generate(x, char_to_idx, idx_to_char, batch_size, num_epochs, vocab_size, window, gru_size, num_layers)
