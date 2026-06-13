import copy
from collections import deque
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)#buffer_sizeだけ入る、いっぱいになると古いデータが自動で消える
        self.batch_size = batch_size
    
    def add(self, state,  action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)
    
    def __len__(self):
        return len(self.buffer)
    
    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)
        #torch.tensorはリストやnumpy配列をニューラルネット用の数値データ構造に変える
        state = torch.tensor(np.stack([x[0] for x in data]))#stackは行列として積む、stateは行列
        action = torch.tensor(np.array([x[1] for x in data]))
        reward = torch.tensor(np.array([x[2] for x in data]))
        next_state = torch.tensor(np.stack([x[3] for x in data]))
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32))
        return state, action, reward, next_state, done
        
    
    
    
        