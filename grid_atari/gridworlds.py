"""
Based on 
https://github.com/automl/TempoRL
"""

import os
import json
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import count
from collections import namedtuple
import time
import numpy as np
from utils import experiments

from envs.grid_envs import GridCore
import gym
from envs.grid_envs import Bridge6x10Env, Pit6x10Env, ZigZag6x10, ZigZag6x10H

device = 'cpu'

def tt(ndarray):
    """
    Helper Function to cast observation to correct type/device
    """
    if device == "cuda":
        return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
    else:
        return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)

def tt_long(ndarray):
    """
    Helper Function to cast observation to correct type/device
    """
    if device == "cuda":
        return Variable(torch.from_numpy(ndarray).long().cuda(), requires_grad=False)
    else:
        return Variable(torch.from_numpy(ndarray).long(), requires_grad=False)

def soft_update(target, source, tau):
    """
    Simple Helper for updating target-network parameters
    :param target: target network
    :param source: policy network
    :param tau: weight to regulate how strongly to update (1 -> copy over weights)
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def get_decay_schedule(start_val: float, decay_start: int, num_steps: int, type_: str):
    """
    Create epsilon decay schedule
    :param start_val: Start decay from this value (i.e. 1)
    :param decay_start: number of iterations to start epsilon decay after
    :param num_steps: Total number of steps to decay over
    :param type_: Which strategy to use. Implemented choices: 'const', 'log', 'linear'
    :return:
    """
    if type_ == 'const':
        return np.array([start_val for _ in range(num_steps)])
    elif type_ == 'log':
        return np.hstack([[start_val for _ in range(decay_start)],
                          np.logspace(np.log10(start_val), np.log10(0.000001), (num_steps - decay_start))])
    elif type_ == 'linear':
        return np.hstack([[start_val for _ in range(decay_start)],
                          np.linspace(start_val, 0, (num_steps - decay_start), endpoint=True)])
    else:
        raise NotImplementedError


class Q(nn.Module):
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

class BoostrappedDQN(nn.Module):
    def __init__(self, state_dim, action_dim, nheads, hidden_dim=50):
        super(BoostrappedDQN, self).__init__()
        self.nheads = nheads
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(state_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, action_dim)) for _ in range(self.nheads)])

    def forward_single_head(self, x, k):
        x = self.heads[k](x)
        return x

    def forward(self, x):
        out = []
        for head in self.heads:
            out.append(head(x))
        return out

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
        return tt_long(batch_states), tt_long(batch_actions), tt_long(batch_next_states), tt(batch_rewards), tt(batch_terminal_flags)

class SkipReplayBuffer:
    """
    Replay Buffer for training the skip-Q.
    Expects "concatenated states" which already contain the behaviour-action for the skip-Q.
    Stores transitions as usual but with additional skip-length. The skip-length is used to properly discount.
    """

    def __init__(self, max_size):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states",
                                                 "rewards", "terminal_flags", "lengths"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[], lengths=[])
        self._size = 0
        self._max_size = max_size

    def add_transition(self, state, action, next_state, reward, done, length):
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.terminal_flags.append(done)
        self._data.lengths.append(length)  # Observed skip-length of the transition
        self._size += 1

        if self._size > self._max_size:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.terminal_flags.pop(0)
            self._data.lengths.pop(0)

    def random_next_batch(self, batch_size):
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_terminal_flags = np.array([self._data.terminal_flags[i] for i in batch_indices])
        batch_lengths = np.array([self._data.lengths[i] for i in batch_indices])
        return tt_long(batch_states), tt_long(batch_actions), tt_long(batch_next_states),\
               tt(batch_rewards), tt(batch_terminal_flags), tt(batch_lengths)

class DQN:
    """
    Simple double DQN Agent
    """
    def __init__(self, state_dim: int, action_dim: int, gamma: float,
                 env: gym.Env):
        """
        Initialize the DQN Agent
        :param state_dim: dimensionality of the input states
        :param action_dim: dimensionality of the output actions
        :param gamma: discount factor
        :param env: environment to train on
        """
        self._q = Q(state_dim, action_dim).to(device)
        self._q_target = Q(state_dim, action_dim).to(device)

        self._gamma = gamma
        self._loss_function = nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.001)
        self._action_dim = action_dim
        self._state_dim = state_dim

        self._replay_buffer = ReplayBuffer(1e6)
        self._env = env

    def get_action(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(tt(x)).detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def train(self, episodes: int, epsilon: float, eval_every: int = 10,
        epsilon_decay: str = "const"):
        """
        Training loop
        :param episodes: maximum number of episodes to train for
        :param max_env_time_steps: maximum number of steps in the environment to perform per episode
        :param epsilon: constant epsilon for exploration when selecting actions
        :param eval_eps: numper of episodes to run for evaluation
        :param eval_every_n_steps: interval of steps after which to evaluate the trained agent
        :param max_train_time_steps: maximum number of steps to train
        :return:
        """
        # Keeps track of episode lengths and rewards
        total_steps = 0
        start_time = time.time()
        epsilon_schedule = get_decay_schedule(epsilon, 0, episodes, epsilon_decay)
        for i_episode in range(episodes):
            s = self._env.reset()
            ed, es, er = 0, 0, 0
            steps, rewards, decisions = [], [], []
            '''if i_episode % eval_every == 0:
                self._env.render(in_control=True)'''
            epsilon = epsilon_schedule[i_episode]

            while True: 
                one_hot_s = np.eye(self._state_dim)[s]
                a = self.get_action(one_hot_s, epsilon)
                ns, r, d, _ = self._env.step(a)

                total_steps += 1
                '''if i_episode % eval_every == 0:
                    self._env.render(in_control=True)'''
                ed += 1
                # Update replay buffer
                self._replay_buffer.add_transition(s, a, ns, r, d)
                batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                    self._replay_buffer.random_next_batch(64)
                er += r
                es += 1
                ########### Begin double Q-learning update
                one_hot_batch_states = F.one_hot(batch_states, num_classes=self._state_dim).float()
                one_hot_batch_next_states = F.one_hot(batch_next_states, num_classes=self._state_dim).float()

                target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                         self._q_target(one_hot_batch_next_states)[torch.arange(64).long(), torch.argmax(
                             self._q(one_hot_batch_next_states), dim=1)]
                current_prediction = self._q(one_hot_batch_states)[torch.arange(64).long(), batch_actions.long()]

                loss = self._loss_function(current_prediction, target.detach())

                self._q_optimizer.zero_grad()
                loss.backward()
                self._q_optimizer.step()

                soft_update(self._q_target, self._q, 0.01)
                ########### End double Q-learning update

                if d:
                    break
                s = ns

            # evaluation    
            steps.append(es)
            rewards.append(er)
            decisions.append(ed)
            eval_stats = dict(
            elapsed_time=time.time() - start_time,
            training_eps=i_episode,
            avg_num_steps_per_ep=float(np.mean(steps)),
            avg_num_decs_per_ep=float(np.mean(decisions)),
            avg_rew_per_ep=float(np.mean(rewards)),
            std_rew_per_ep=float(np.std(rewards))
            )
            print('Done %4d/%4d episodes, rewards: %4d' % (i_episode, episodes, float(np.mean(rewards))))
            with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                json.dump(eval_stats, out_fh)
                out_fh.write('\n')

    def save(self, filename):
        torch.save(self._q.state_dict(), filename + "_DQN")
        torch.save(self._q_optimizer.state_dict(), filename + "_DQN_optimizer")

    def load(self, filename):
        self._q.load_state_dict(torch.load(filename + "_DQN"))
        self._q_optimizer.load_state_dict(torch.load(filename + "_DQN_optimizer"))

class TDQN:
    def __init__(self, state_dim: int, action_dim: int, skip_dim: int, gamma: float,
                 env: gym.Env):

        self._q = Q(state_dim, action_dim).to(device)
        self._q_target = Q(state_dim, action_dim).to(device)
        self._skip_q = Q(state_dim + 1, skip_dim).to(device)

        self._gamma = gamma
        self._loss_function = nn.MSELoss()
        self._skip_loss_function = nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.001)
        self._skip_q_optimizer = optim.Adam(self._skip_q.parameters(), lr=0.001)
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._skip_dim = skip_dim

        self._replay_buffer = ReplayBuffer(1e6)
        self._skip_replay_buffer = SkipReplayBuffer(1e6)
        self._env = env

    def get_action(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(tt(x)).detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def get_skip(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get the skip epsilon-greedy based on observation x
        """
        u = np.argmax(self._skip_q(tt(x)).detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._skip_dim)
        return u

    def train(self, episodes: int, epsilon: float, eval_every: int = 10,
        epsilon_decay: str = "const"):
        """
        Training loop
        :param episodes: maximum number of episodes to train for
        :param epsilon: constant epsilon for exploration when selecting actions
        :param eval_eps: numper of episodes to run for evaluation
        :param eval_every_n_steps: interval of steps after which to evaluate the trained agent
        :return:
        """
        epsilon_schedule_action = get_decay_schedule(epsilon, 0, episodes, epsilon_decay)
        epsilon_schedule_temporal = get_decay_schedule(epsilon, 0, episodes, epsilon_decay)
        start_time = time.time()
        for i_episode in range(episodes):
            s = self._env.reset()
            ed, es, er = 0, 0, 0
            steps, rewards, decisions = [], [], []
            epsilon_action = epsilon_schedule_action[i_episode]
            epsilon_temporal = epsilon_schedule_temporal[i_episode]
            while True:
                one_hot_s = np.eye(self._state_dim)[s]
                a = self.get_action(one_hot_s, epsilon_action)          
                skip_state = np.hstack([one_hot_s, [a]])  # concatenate action to the state
                skip = self.get_skip(skip_state, epsilon_temporal)
                d = False
                ed += 1
                skip_states, skip_rewards = [], []
                for curr_skip in range(skip + 1):  # play the same action a "skip" times
                    ns, r, d, _ = self._env.step(a)
                    er += r
                    es += 1
                    one_hot_s = np.eye(self._state_dim)[s]
                    skip_states.append(np.hstack([one_hot_s, [a]]))  # keep track of all states that are visited inbetween
                    skip_rewards.append(r)

                    # Update the skip buffer with all observed transitions
                    skip_id = 0
                    for start_state in skip_states:
                        skip_reward = 0
                        for exp, r in enumerate(skip_rewards[skip_id:]):  # make sure to properly discount
                            skip_reward += np.power(self._gamma, exp) * r

                        self._skip_replay_buffer.add_transition(start_state, curr_skip - skip_id, ns,
                                                                skip_reward, d, curr_skip - skip_id + 1) 
                        skip_id += 1

                   # Skip Q update based on double DQN where the target is the behaviour network
                    batch_states, batch_actions, batch_next_states, batch_rewards, \
                    batch_terminal_flags, batch_lengths = self._skip_replay_buffer.random_next_batch(64)
                    
                    one_hot_batch_next_states = F.one_hot(batch_next_states, num_classes=self._state_dim).float()

                    target = batch_rewards + (1 - batch_terminal_flags) * torch.pow(self._gamma, batch_lengths) * \
                             self._q_target(one_hot_batch_next_states)[torch.arange(64).long(), torch.argmax(
                                 self._q(one_hot_batch_next_states), dim=1)]
                    current_prediction = self._skip_q(batch_states.float())[torch.arange(64).long(), batch_actions.long()]

                    loss = self._skip_loss_function(current_prediction, target.detach())

                    self._skip_q_optimizer.zero_grad()
                    loss.backward()
                    self._skip_q_optimizer.step()

                    # Update replay buffer
                    self._replay_buffer.add_transition(s, a, ns, r, d)
                    batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                        self._replay_buffer.random_next_batch(64)
    
                    ########### Begin double Q-learning update
                    one_hot_batch_states = F.one_hot(batch_states, num_classes=self._state_dim).float()
                    one_hot_batch_next_states = F.one_hot(batch_next_states, num_classes=self._state_dim).float()

                    target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                            self._q_target(one_hot_batch_next_states)[torch.arange(64).long(), torch.argmax(
                                self._q(one_hot_batch_next_states), dim=1)]
                    current_prediction = self._q(one_hot_batch_states)[torch.arange(64).long(), batch_actions.long()]

                    loss = self._loss_function(current_prediction, target.detach())

                    self._q_optimizer.zero_grad()
                    loss.backward()
                    self._q_optimizer.step()

                    soft_update(self._q_target, self._q, 0.01)
                    ########### End double Q-learning update
                    if d:
                        break
                    s = ns
                if d:
                    break

            # evaluation    
            steps.append(es)
            rewards.append(er)
            decisions.append(ed)
            eval_stats = dict(
            elapsed_time=time.time() - start_time,
            training_eps=i_episode,
            avg_num_steps_per_ep=float(np.mean(steps)),
            avg_num_decs_per_ep=float(np.mean(decisions)),
            avg_rew_per_ep=float(np.mean(rewards)),
            std_rew_per_ep=float(np.std(rewards))
            )
            print('Done %4d/%4d episodes, rewards: %4d' % (i_episode, episodes, float(np.mean(rewards))))
            with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                json.dump(eval_stats, out_fh)
                out_fh.write('\n')

            if float(np.mean(rewards)) >= 1.0 and (i_episode % 10) == 0:
                self.save(os.path.join(out_dir, 'model'))

    def save(self, filename):
        torch.save(self._q.state_dict(), filename + "_TDQN")
        torch.save(self._q_optimizer.state_dict(), filename + "_TDQN_optimizer")
        torch.save(self._skip_q.state_dict(), filename + "_TDQN_skip")
        torch.save(self._skip_q_optimizer.state_dict(), filename + "_TDQN_skip_optimizer")

    def load(self, filename):
        self._q.load_state_dict(torch.load(filename + "_TDQN"))
        self._q_optimizer.load_state_dict(torch.load(filename + "_TDQN_optimizer"))
        self._skip_q.load_state_dict(torch.load(filename + "_TDQN_skip"))
        self._skip_q_optimizer.load_state_dict(torch.load(filename + "_TDQN_skip_optimizer"))

class UTE:
    def __init__(self, state_dim: int, action_dim: int, skip_dim: int, uncertainty_factor:float, gamma: float,
                 env: gym.Env):

        self._q = Q(state_dim, action_dim).to(device)
        self._q_target = Q(state_dim, action_dim).to(device)
        self.n_heads = 10
        self._skip_q = BoostrappedDQN(state_dim + 1, skip_dim, self.n_heads).to(device)

        self._gamma = gamma
        self._loss_function = nn.MSELoss()
        self._skip_loss_function = nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.001)
        self._skip_q_optimizer = optim.Adam(self._skip_q.parameters(), lr=0.001)
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._skip_dim = skip_dim

        self.bernoulli_probability = 0.5
        self.uncertainty_factor = uncertainty_factor

        self._replay_buffer = ReplayBuffer(1e6)
        self._skip_replay_buffer = SkipReplayBuffer(1e6)
        self._env = env

    def get_action(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(tt(x)).detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def get_skip(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get the skip epsilon-greedy based on observation x
        """
        current_outputs = self._skip_q(tt(x))
        outputs = []
        for k in range(self.n_heads):
            outputs.append(current_outputs[k].detach().cpu().numpy())
        outputs = np.array(outputs)
        mean_Q = np.mean(outputs , axis=0) # 1x10
        std_Q = np.std(outputs, axis=0)
        Q_tilda = mean_Q + self.uncertainty_factor*std_Q
        u = np.argmax(Q_tilda)
        return u

    def train(self, episodes: int, epsilon: float, eval_every: int = 10,
        epsilon_decay: str = "const"):
        """
        Training loop
        :param episodes: maximum number of episodes to train for
        :param epsilon: constant epsilon for exploration when selecting actions
        :param eval_eps: numper of episodes to run for evaluation
        :param eval_every_n_steps: interval of steps after which to evaluate the trained agent
        :return:
        """
        batch_size=32
        epsilon_schedule_action = get_decay_schedule(epsilon, 0, episodes, epsilon_decay)
        start_time = time.time()
        for i_episode in range(episodes):
            s = self._env.reset()
            ed, es, er = 0, 0, 0
            steps, rewards, decisions = [], [], []
            epsilon_action = epsilon_schedule_action[i_episode]
            while True:
                one_hot_s = np.eye(self._state_dim)[s]
                a = self.get_action(one_hot_s, epsilon_action)          
                skip_state = np.hstack([one_hot_s, [a]])  # concatenate action to the state
                skip = self.get_skip(skip_state, 0)
                d = False
                ed += 1
                skip_states, skip_rewards = [], []
                
                for curr_skip in range(skip + 1):  # play the same action a "skip" times
                    ns, r, d, _ = self._env.step(a)
                    er += r
                    es += 1
                    one_hot_s = np.eye(self._state_dim)[s]
                    skip_states.append(np.hstack([one_hot_s, [a]]))  # keep track of all states that are visited inbetween
                    skip_rewards.append(r)

                    # Update the skip buffer with all observed transitions
                    skip_id = 0
                    for start_state in skip_states:
                        skip_reward = 0
                        for exp, r in enumerate(skip_rewards[skip_id:]):  # make sure to properly discount
                            skip_reward += np.power(self._gamma, exp) * r

                        self._skip_replay_buffer.add_transition(start_state, curr_skip - skip_id, ns,
                                                                skip_reward, d, curr_skip - skip_id + 1) 
                        skip_id += 1

                   # Skip Q update based on double DQN where the target is the behaviour network
                    batch_states, batch_actions, batch_next_states, batch_rewards, \
                    batch_terminal_flags, batch_lengths = self._skip_replay_buffer.random_next_batch(batch_size*2)
                    
                    one_hot_batch_next_states = F.one_hot(batch_next_states, num_classes=self._state_dim).float()

                    target = batch_rewards + (1 - batch_terminal_flags) * torch.pow(self._gamma, batch_lengths) * \
                             self._q_target(one_hot_batch_next_states)[torch.arange(batch_size*2).long(), torch.argmax(
                                 self._q(one_hot_batch_next_states), dim=1)]

                    current_outputs = self._skip_q(batch_states.float())
                    masks = torch.bernoulli(torch.zeros((batch_size*2, self.n_heads), device=device) + self.bernoulli_probability )
                    cnt_losses = []
                    for k in range(self.n_heads):
                        total_used = torch.sum(masks[:,k])
                        if total_used > 0.0:
                            current_prediction = current_outputs[k][torch.arange(batch_size*2).long(), batch_actions.long()]
                            l1loss = self._skip_loss_function(current_prediction, target.detach())
                            full_loss = masks[:,k]*l1loss
                            loss = torch.sum(full_loss/total_used)
                            cnt_losses.append(loss)

                    self._skip_q_optimizer.zero_grad()
                    skip_loss = sum(cnt_losses)/self.n_heads
                    skip_loss.backward()
                    self._skip_q_optimizer.step()


                    # Update replay buffer
                    self._replay_buffer.add_transition(s, a, ns, r, d)
                    batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                        self._replay_buffer.random_next_batch(batch_size)
    
                    ########### Begin double Q-learning update
                    one_hot_batch_states = F.one_hot(batch_states, num_classes=self._state_dim).float()
                    one_hot_batch_next_states = F.one_hot(batch_next_states, num_classes=self._state_dim).float()

                    target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                            self._q_target(one_hot_batch_next_states)[torch.arange(batch_size).long(), torch.argmax(
                                self._q(one_hot_batch_next_states), dim=1)]
                    current_prediction = self._q(one_hot_batch_states)[torch.arange(batch_size).long(), batch_actions.long()]

                    loss = self._loss_function(current_prediction, target.detach())

                    self._q_optimizer.zero_grad()
                    loss.backward()
                    self._q_optimizer.step()

                    soft_update(self._q_target, self._q, 0.01)
                    ########### End double Q-learning update

                    if d:
                        break
                    s = ns
                if d:
                    break

            # evaluation    
            steps.append(es)
            rewards.append(er)
            decisions.append(ed)
            eval_stats = dict(
            elapsed_time=time.time() - start_time,
            training_eps=i_episode,
            avg_num_steps_per_ep=float(np.mean(steps)),
            avg_num_decs_per_ep=float(np.mean(decisions)),
            avg_rew_per_ep=float(np.mean(rewards)),
            std_rew_per_ep=float(np.std(rewards))
            )
            print('Done %4d/%4d episodes, rewards: %4d' % (i_episode, episodes, float(np.mean(rewards))))
            
            with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                json.dump(eval_stats, out_fh)
                out_fh.write('\n')
  
    def save(self, filename):
        torch.save(self._q.state_dict(), filename + "_UTE")
        torch.save(self._q_optimizer.state_dict(), filename + "_UTE_optimizer")
        torch.save(self._skip_q.state_dict(), filename + "_UTE_skip")
        torch.save(self._skip_q_optimizer.state_dict(), filename + "_UTE_skip_optimizer")

    def load(self, filename):
        self._q.load_state_dict(torch.load(filename + "_UTE"))
        self._q_optimizer.load_state_dict(torch.load(filename + "_UTE_optimizer"))
        self._skip_q.load_state_dict(torch.load(filename + "_UTE_skip"))
        self._skip_q_optimizer.load_state_dict(torch.load(filename + "_UTE_skip_optimizer"))



if __name__ == '__main__':
    import argparse

    outdir_suffix_dict = {'none': '', 'empty': '', 'time': '%Y%m%dT%H%M%S.%f',
                          'seed': '{:d}', 'params': '{:d}_{:d}_{:d}',
                          'paramsseed': '{:d}_{:d}_{:d}_{:d}'}
    parser = argparse.ArgumentParser('Skip-MDP Tabular-Q')
    parser.add_argument('--episodes', '-e',
                        default=1000,
                        type=int,
                        help='Number of training episodes')
    parser.add_argument('--out-dir',
                        default='experiments/tabular/',
                        type=str,
                        help='Directory to save results. Defaults to tmp dir.')
    parser.add_argument('--out-dir-suffix',
                        default='paramsseed',
                        type=str,
                        choices=list(outdir_suffix_dict.keys()),
                        help='Created suffix of directory to save results.')
    parser.add_argument('--seed', '-s',
                        default=42,
                        type=int,
                        help='Seed')
    parser.add_argument('--env-max-steps',
                        default=50,
                        type=int,
                        help='Maximal steps in environment before termination.',
                        dest='env_ms')
    parser.add_argument('--agent-eps-decay',
                        default='log',
                        choices={'linear', 'log', 'const'},
                        help='Epsilon decay schedule',
                        dest='agent_eps_d')
    parser.add_argument('--agent-eps',
                        default=1.0,
                        type=float,
                        help='Epsilon value. Used as start value when decay linear or log. Otherwise constant value.',
                        dest='agent_eps')
    parser.add_argument('--agent',
                        default='sq',
                        choices={'sq', 'q', 'ute'},
                        type=str.lower,
                        help='Agent type to train')
    parser.add_argument('--env',
                        default='lava',
                        choices={'lava', 'lava2',
                                 'lava_perc', 'lava2_perc',
                                 'lava_ng', 'lava2_ng',
                                 'lava3', 'lava3_perc', 'lava3_ng'},
                        type=str.lower,
                        help='Enironment to use')
    parser.add_argument('--eval-eps',
                        default=10,
                        type=int,
                        help='After how many episodes to evaluate')
    parser.add_argument('--stochasticity',
                        default=0.0,
                        type=float,
                        help='probability of the selected action failing and instead executing any of the remaining 3')
    parser.add_argument('--no-render',
                        action='store_true',
                        help='Deactivate rendering of environment evaluation')
    parser.add_argument('--max-skips',
                        type=int,
                        default=7,
                        help='Max skip size for tempoRL')
    parser.add_argument('--uncertainty-factor',
                        type=float,
                        default=-1.5,
                        help='for uncertainty-sensitive model')    

    # setup output dir
    args = parser.parse_args()
    outdir_suffix_dict['seed'] = outdir_suffix_dict['seed'].format(args.seed)
    outdir_suffix_dict['params'] = outdir_suffix_dict['params'].format(
        args.episodes, args.max_skips, args.env_ms)
    outdir_suffix_dict['paramsseed'] = outdir_suffix_dict['paramsseed'].format(
        args.episodes, args.max_skips, args.env_ms, args.seed)

    if not args.no_render:
        # Clear screen in ANSI terminal
        print('\033c')
        print('\x1bc')
    
    if args.agent == 'ute':
        _out_dir = args.out_dir+'/'+args.env+f'_{args.stochasticity}'+'/'+args.agent+'/'+args.agent_eps_d+'/uncertainty_factor'+str(args.uncertainty_factor)
    else:
        _out_dir = args.out_dir+'/'+args.env+f'_{args.stochasticity}'+'/'+args.agent+'/'+args.agent_eps_d
    out_dir = experiments.prepare_output_dir(args, user_specified_dir=_out_dir,
                                             time_format=outdir_suffix_dict[args.out_dir_suffix]) 
    
    np.random.seed(args.seed)  # seed nump
    env = None

    if args.env.startswith('lava'):

        perc = args.env.endswith('perc')
        ng = args.env.endswith('ng')
        if args.env.startswith('lava2'):
            env = Bridge6x10Env(max_steps=args.env_ms, percentage_reward=perc, no_goal_rew=ng,
                              act_fail_prob=args.stochasticity, numpy_state=False)
        elif args.env.startswith('lava3'):
            env = ZigZag6x10(max_steps=args.env_ms, percentage_reward=perc, no_goal_rew=ng, goal=(5, 9),
                           act_fail_prob=args.stochasticity, numpy_state=False)
        elif args.env.startswith('lava4'):
            env = ZigZag6x10H(max_steps=args.env_ms, percentage_reward=perc, no_goal_rew=ng, goal=(5, 9),
                            act_fail_prob=args.stochasticity, numpy_state=False)
        else:
            env = Pit6x10Env(max_steps=args.env_ms, percentage_reward=perc, no_goal_rew=ng,
                           act_fail_prob=args.stochasticity, numpy_state=False)

    # setup agent
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    if args.agent == 'sq':
        agent = TDQN(state_dim, action_dim, args.max_skips, gamma=0.99, env=env)
    elif args.agent == 'q':
        agent = DQN(state_dim, action_dim, gamma=0.99, env=env)
    elif args.agent == 'ute':
        agent = UTE(state_dim, action_dim, args.max_skips, args.uncertainty_factor, gamma=0.99, env=env)   
    else:
        raise NotImplemented
    
    episodes = args.episodes
    max_env_time_steps = args.env_ms
    epsilon = 0.1

    agent.train(episodes, epsilon, eval_every=args.eval_eps, epsilon_decay=args.agent_eps_d)
    file_name = f"{args.agent}_{args.env}_{args.seed}"
    #agent.save(f"{out_dir}/{file_name}")