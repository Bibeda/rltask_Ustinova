import torch
from collections import deque
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random

class BufferMemory():

    def __init__(self,memsize, time_step):
        self.memsize = memsize
        self.memory = deque(maxlen=self.memsize)
        self.time_step = time_step

    def push(self,episode):
        if len(episode) >= self.time_step:
            self.memory.append(episode)

    def get_batch(self, bsize):
        sampled_episodes = random.sample(self.memory, bsize)

        states = []
        actions = []
        rewards = []
        next_states = []
        truncated_batch = []
        terminated_batch = []

        for episode in sampled_episodes:
            start = np.random.randint(0, len(episode) - self.time_step + 1)
            subseq = episode[start:start + self.time_step]

            states.append([step[0] for step in subseq])
            actions.append([step[1] for step in subseq])
            rewards.append([step[2] for step in subseq])
            next_states.append([step[3] for step in subseq])
            terminated_batch.append([step[4] for step in subseq])
            truncated_batch.append([step[4] for step in subseq])

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        terminated_batch = torch.tensor(np.array(terminated_batch), dtype=torch.bool)
        truncated_batch = torch.tensor(np.array(truncated_batch), dtype=torch.bool)

        return states, actions, rewards, next_states, terminated_batch, truncated_batch

    def __len__(self):
        return len(self.memory)    



class RecurrentDQN (nn.Module):
    def __init__(self, input_size, out_size, lstm_hidden=128):
        super().__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.lstm_hidden = lstm_hidden

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        self.q_head = nn.Linear(lstm_hidden, out_size)

    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x, hidden)
        out = lstm_out[:, -1, :]
        q_vals = self.q_head(out)
        return q_vals, hidden

    def init_hidden_states(self, bsize, device='cpu'):
        h = torch.zeros(1, bsize, self.lstm_hidden, device=device)
        c = torch.zeros(1, bsize, self.lstm_hidden, device=device)
        return (h, c)
    
