# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn

from commun.utils import initialize_weights
from commun.noisy_layer import NoisyLinear
import numpy as np
from torch.distributions import Categorical


class DQN(nn.Module):
    def __init__(self, args, action_space):
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Linear(args.input_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True)
        )
        self.last_layer = nn.Linear(16, action_space)
        initialize_weights(self)

    def forward(self, x):
        x = self.features(x)
        x = self.last_layer(x)
        return x


class TabularQ():
    def __init__(self, args, action_space):
        self.Q = np.full((args.input_dim, action_space), 5.0)
    
    # def get_Qvals(self, state):
    #     state = np.where(state == 1)[0]
    #     return self.Q[state, :]

    def update_Qval(self, state, action, reward, new_state, done, lr, gamma):

        state = np.where(state == 1)[0][0]
        new_state = np.where(new_state == 1)[0][0]

        if not done:
            self.Q[state, action] = self.Q[state, action] + lr * (reward + gamma * np.max(self.Q[new_state, :]) - self.Q[state, action])
        else:
            self.Q[state, action] = self.Q[state, action] + lr * (reward - self.Q[state, action])

        # if state == 9 or state == 8:
        #     print("================")
        #     print(state, action, done)
        #     print(self.Q)


        
class BoostrappedDQN(nn.Module):
    def __init__(self, args, action_space):
        nn.Module.__init__(self)
        # self.features = nn.Sequential(
        #     nn.Linear(args.input_dim, 16),
        #     nn.ReLU(inplace=True)
        # )
        self.nheads = args.nheads
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(args.input_dim, 16),
        nn.ReLU(inplace=True),
        nn.Linear(16, 16),
        nn.ReLU(inplace=True),
        nn.Linear(16, action_space)) for _ in range(args.nheads)])

        initialize_weights(self)

    def forward_single_head(self, x, k):
        # x = self.features(x)
        x = self.heads[k](x)
        return x

    def forward(self, x):
        # x = self.features(x)
        out = []
        for head in self.heads:
            out.append(head(x))
        return out


class NoisyDQN(nn.Module):
    def __init__(self, args, action_space):
        nn.Module.__init__(self)
        self.fc1 = NoisyLinear(args.input_dim+2, 16)
        self.fc2 = NoisyLinear(16, 16)
        self.fc3 = NoisyLinear(16, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()

