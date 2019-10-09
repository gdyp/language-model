#! -*- coding: utf-8 -*-
"""
dnn 语言模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class DNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, embedding=None):
        super(DNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)

        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq):
        # batch_size, seq_len = input_seq.shape
        input_embedding = self.embedding(input_seq)
        # packed = pack_padded_sequence(input_embedding, input_length, batch_first=True)
        output, _ = self.lstm(input_embedding)
        # unpacked, _ = pad_packed_sequence(output, batch_first=True)

        last_hidden_state = output[:, -1, :].squeeze()
        output = self.linear(last_hidden_state)
        return last_hidden_state, output
