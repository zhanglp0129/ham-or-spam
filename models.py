import torch
import numpy as np
from torch import nn, optim
import conf
import data

class HamSpamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(data.dictionary), conf.embedding_size)
        self.lstm = nn.LSTM(
            input_size=conf.embedding_size,
            hidden_size=conf.hidden_size,
            num_layers=conf.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=conf.dropout
        )
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(conf.hidden_size * 2, conf.hidden_size),
            nn.ReLU(),
            nn.Dropout(conf.dropout),
            nn.BatchNorm1d(conf.hidden_size),
            nn.Linear(conf.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embedding(x)
        out, (h_n, c_n) = self.lstm(x)
        x = torch.concat([h_n[-2,:,:],h_n[-1,:,:]],dim=-1)
        return self.fc(x)

if __name__ == "__main__":
    model = HamSpamModel()
    print(model)