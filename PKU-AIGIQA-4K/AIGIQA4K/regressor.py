# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, args, hidden_size):
        super(MLP, self).__init__()
        if args.benchmark == 'I2I' and args.using_prompt == 1:
            self.hidden_size = hidden_size * 2
        else:
            self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.fc2 = nn.Linear(self.hidden_size // 2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x