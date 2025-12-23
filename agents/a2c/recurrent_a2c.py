import torch
from collections import deque
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
    
class RecurrentA2C(nn.Module):

    def __init__(self, input_dim, output_dim, lstm_hidden):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.fc1 = nn.Linear(input_dim, lstm_hidden)
        
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            batch_first=True
        )
        self.fc_actor = nn.Linear(lstm_hidden, output_dim)
        self.fc_critic = nn.Linear(lstm_hidden, 1)

    def forward(self, x, hidden=None):
        x = torch.relu(self.fc1(x))
        lstm_out, hidden = self.lstm(x, hidden)
        last_out = lstm_out[:, -1, :]
        action_probs = torch.softmax(self.fc_actor(last_out), dim=-1)
        state_value = self.fc_critic(last_out)
        return action_probs, state_value, hidden
    
    def init_hidden_states(self, bsize):
        h = torch.zeros(1, bsize, self.lstm_hidden)
        c = torch.zeros(1, bsize, self.lstm_hidden)
        return (h, c)
