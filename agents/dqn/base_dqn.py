import torch
from collections import deque
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# from DQNTrainer import DQNTrainer
# from GridWorldEnv import GridWorldEnv

class ReplayMemory():
    def __init__(self, memsize):
        self.memory = deque([], maxlen=memsize)

    def push(self, state, action, next_state, reward, terminated, truncated):
        self.memory.append((state, action, next_state, reward, terminated, truncated))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]

        states, actions, next_states, rewards, terminated, truncated = zip(*batch)
        states = torch.cat(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        next_states = torch.cat(next_states)
        rewards = torch.FloatTensor(rewards)
        terminated_batch = torch.FloatTensor(terminated)
        truncated_batch = torch.FloatTensor(truncated)

        return states, actions, next_states, rewards, terminated_batch, truncated_batch

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
