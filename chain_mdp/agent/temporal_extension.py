# baseline code copied from https://github.com/automl/TempoRL

import os
import json
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import count
from collections import namedtuple
from collections import Counter
import time


device = torch.device('cpu')

def tt(ndarray):
    """
    Helper Function to cast observation to correct type/device
    """
    if device == "cuda":
        return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
    else:
        return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)

def soft_update(target, source, tau):
    """
    Simple Helper for updating target-network parameters
    :param target: target network
    :param source: policy network
    :param tau: weight to regulate how strongly to update (1 -> copy over weights)
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    """
    See soft_update
    """
    soft_update(target, source, 1.0)


class TQ(nn.Module):
    """
    Q-Function that takes the behaviour action as context.
    This Q is solely inteded to be used for computing the skip-Q Q(s,j|a).

    Basically the same architecture as Q but with context input layer.
    """

    def __init__(self, state_dim, skip_dim, non_linearity=F.relu, hidden_dim=16):
        super(TQ, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.skip_fc2 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.skip_fc3 = nn.Linear(hidden_dim + 10, skip_dim)  # output layer taking context and state into account
        self._non_linearity = non_linearity

    def forward(self, x, a=None):
        # Process the input state
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))

        # Process behaviour-action as context
        x_ = self._non_linearity(self.skip_fc2(a))
        return self.skip_fc3(torch.cat([x, x_], -1))  # Concatenate both to produce the final output


class Q(nn.Module):
    """
    Simple fully connected Q function. Also used for skip-Q when concatenating behaviour action and state together.
    Used for simpler environments such as mountain-car or lunar-lander.
    """

    def __init__(self, state_dim, action_dim, non_linearity=F.relu, hidden_dim=50):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self._non_linearity = non_linearity

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """
    Simple Replay Buffer. Used for standard DQN learning.
    """

    def __init__(self, max_size):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "terminal_flags"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[])
        self._size = 0
        self._max_size = max_size

    def add_transition(self, state, action, next_state, reward, done):
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.terminal_flags.append(done)
        self._size += 1

        if self._size > self._max_size:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.terminal_flags.pop(0)

    def random_next_batch(self, batch_size):
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_terminal_flags = np.array([self._data.terminal_flags[i] for i in batch_indices])
        return tt(batch_states), tt(batch_actions), tt(batch_next_states), tt(batch_rewards), tt(batch_terminal_flags)


class NoneConcatSkipReplayBuffer:
    """
    Replay Buffer for training the skip-Q.
    Expects states in which the behaviour-action is not siply concatenated for the skip-Q.
    Stores transitions as usual but with additional skip-length. The skip-length is used to properly discount.
    Additionally stores the behaviour_action which is the context for this skip-transition.
    """

    def __init__(self, max_size):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states",
                                                 "rewards", "terminal_flags", "lengths", "behaviour_action"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[], lengths=[],
                                behaviour_action=[])
        self._size = 0
        self._max_size = max_size

    def add_transition(self, state, action, next_state, reward, done, length, behaviour):
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.terminal_flags.append(done)
        self._data.lengths.append(length)  # Observed skip-length
        self._data.behaviour_action.append(behaviour)  # Behaviour action to condition skip on
        self._size += 1

        if self._size > self._max_size:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.terminal_flags.pop(0)
            self._data.lengths.pop(0)
            self._data.behaviour_action.pop(0)

    def random_next_batch(self, batch_size):
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_terminal_flags = np.array([self._data.terminal_flags[i] for i in batch_indices])
        batch_lengths = np.array([self._data.lengths[i] for i in batch_indices])
        batch_behavoiurs = np.array([self._data.behaviour_action[i] for i in batch_indices])
        return tt(batch_states), tt(batch_actions), tt(batch_next_states), \
               tt(batch_rewards), tt(batch_terminal_flags), tt(batch_lengths), tt(batch_behavoiurs)



class TDQN:
    """
    TempoRL DQN agent capable of handling more complex state inputs through use of contextualized behaviour actions.
    """

    def __init__(self, state_dim, action_dim, skip_dim, gamma):
        """
        Initialize the TDQN Agent
        :param state_dim: dimensionality of the input states
        :param action_dim: dimensionality of the action output
        :param skip_dim: dimenionality of the skip output
        :param gamma: discount factor
        :param env: environment to train on
        :param eval_env: environment to evaluate on
        :param vision: boolean flag to indicate if the input state is an image or not
        :param shared: boolean flag to indicate if a weight sharing input representation is used or not.
        """

        self._q = Q(state_dim, action_dim).to(device)
        self._q_target = Q(state_dim, action_dim).to(device)
        self._skip_q = TQ(state_dim, skip_dim).to(device)
        print('Using {} as Q'.format(str(self._q)))
        print('Using {} as skip-Q\n{}'.format(str(self._skip_q), '#' * 80))

        self._gamma = gamma
        self._loss_function = nn.MSELoss()
        self._skip_loss_function = nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.001)
        self._skip_q_optimizer = optim.Adam(self._skip_q.parameters(), lr=0.001)
        self._action_dim = action_dim
        self._skip_dim = skip_dim

        self._replay_buffer = ReplayBuffer(1e6)
        self._skip_replay_buffer = NoneConcatSkipReplayBuffer(1e6)

    def get_action(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(tt(x).to(device)).detach().cpu().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def get_skip(self, x: np.ndarray, a: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get the skip epsilon-greedy based on observation x conditioned on behaviour action a
        """
        u = np.argmax(self._skip_q(tt(x).to(device), tt(a).to(device)).detach().cpu().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._skip_dim)
        return u

class EnsembleNet(nn.Module):
    def __init__(self, state_dim, skip_dim, nheads=10, hidden_dim=16):
        nn.Module.__init__(self)
        self.nheads = nheads
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(state_dim+1, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, skip_dim)) for _ in range(nheads)])

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
    

class UTE:
    """
    Uncertainty-aware Temporal Extension
    """

    def __init__(self, state_dim, action_dim, skip_dim,  gamma, env=None, uncertainty_factor=0.0):
        """
        Initialize the UTE Agent
        :param state_dim: dimensionality of the input states
        :param action_dim: dimensionality of the action output
        :param skip_dim: dimenionality of the skip output
        :param gamma: discount factor
        :param env: environment to train on
        :param eval_env: environment to evaluate on
        :param vision: boolean flag to indicate if the input state is an image or not
        :param shared: boolean flag to indicate if a weight sharing input representation is used or not.
        :param uncertainty_factor: float for uncertainty-senstivity parameter
        """

        self.core_net = Q(state_dim, action_dim).to(device)
        self._q = Q(state_dim, action_dim).to(device)
        self._q_target = Q(state_dim, action_dim).to(device)
        #hard_update(self._q_target, self._q)

        self.n_heads = 10
        self._skip_q = EnsembleNet(state_dim, skip_dim).to(device)
        print('Using {} as Q'.format(str(self._q)))
        print('Using {} as skip-Q\n{}'.format(str(self._skip_q), '#' * 80))

        self._gamma = gamma
        self._action_dim = action_dim
        self._skip_dim = skip_dim

        self.batch_size = 32
        self.target_net_upd_freq = 500
        self.learning_starts = 10_000
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.01
        self.epsilon_timesteps = 200_000
        self.train_freq = 4
        self.bernoulli_probability = 0.5
        self.uncertainty_factor = uncertainty_factor
        print(f"uncertainty factor is {uncertainty_factor}")

        self._loss_function = nn.SmoothL1Loss()  # huber loss # nn.MSELoss()
        self._skip_loss_function = nn.SmoothL1Loss()  # nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
        self._skip_q_optimizer = optim.Adam(self._skip_q.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
        
        self._replay_buffer = ReplayBuffer(5e4)
        self._skip_replay_buffer = NoneConcatSkipReplayBuffer(5e4)

        self.env = env


    def get_action(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(tt(x[None, :])).cpu().detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def get_skip(self, x , a):
        skip_state = np.hstack([x, [a]])  # concatenate action to the state
        current_outputs = self._skip_q(tt(skip_state))
        outputs = []
        for k in range(self.n_heads):
            outputs.append(current_outputs[k].detach().cpu().numpy())
        outputs = np.array(outputs)
        #norms = np.linalg.norm(outputs, axis=-1, keepdims=True) # shape= Hx1x10
        #norms = np.repeat(norms, self._skip_dim, axis=-1)
        #norm_outputs = outputs/(norms + 0.0000001) 
        mean_Q = np.mean(outputs, axis=0) # 1x10
        std_Q = np.std(outputs, axis=0)
        Q_tilda = mean_Q + self.uncertainty_factor*std_Q
        if Q_tilda.shape[0] > 1 :
            u = np.argmax(Q_tilda, axis=-1)
        else:
            u = np.argmax(Q_tilda)
        return u

