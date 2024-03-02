"""
Based on 
https://github.com/automl/TempoRL
"""

import os
import json
import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
from collections import namedtuple
import time
from utils import experiments
from utils.ucb import UCB
from utils.atari_utils import tt, soft_update, hard_update, atari_initializer
from utils.env_wrappers import make_env, make_env_old
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NatureDQN(nn.Module):
    """
    DQN following the DQN implementation from
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    """

    def __init__(self, in_channels=4, num_actions=18):
        """
        :param in_channels: number of channel of input. (how many stacked images are used)
        :param num_actions: action values
        """
        super(NatureDQN, self).__init__()
        if env.observation_space.shape[-1] == 84: #hack
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        elif env.observation_space.shape[-1] == 42: #hack
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2)
        else:
            raise ValueError("Check state space dimensionality. Expected nx42x42 or nx84x84. Was:", env.observation_space.shape)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.reshape(x.size(0), -1)))
        return self.fc5(x)

    def get_feature_rep(self, x):
        # Process input image
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        return x

class NatureWeightsharingTQN(nn.Module):
    """
    Network to learn the skip behaviour using the same architecture as the original DQN but with additional context.
    The context is expected to be the chosen behaviour action on which the skip-Q is conditioned.
    This implementation allows to share weights between the behaviour network and the skip network
    """

    def __init__(self, in_channels=4, num_actions=18, num_skip_actions=10):
        """
        :param in_channels: number of channel of input. (how many stacked images are used)
        :param num_actions: action values
        """
        super(NatureWeightsharingTQN, self).__init__()
        # shared input-layers
        if env.observation_space.shape[-1] == 84: #hack
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        elif env.observation_space.shape[-1] == 42: #hack
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2)
        else:
            raise ValueError("Check state space dimensionality. Expected nx42x42 or nx84x84. Was:", env.observation_space.shape)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # skip-layers
        self.skip = nn.Linear(1, 10)  # Context layer
        self.skip_fc4 = nn.Linear(7 * 7 * 64 + 10, 512)
        self.skip_fc5 = nn.Linear(512, num_skip_actions)

        # behaviour-layers
        self.action_fc4 = nn.Linear(7 * 7 * 64, 512)
        self.action_fc5 = nn.Linear(512, num_actions)

    def forward(self, x, action_val=None):
        # Process input image
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        if action_val is not None:  # Only if an action_value was provided we evaluate the skip output layers Q(s,j|a)
            x_ = F.relu(self.skip(action_val))
            x = F.relu(self.skip_fc4(
                torch.cat([x.reshape(x.size(0), -1), x_], 1)))  # This layer concatenates the context and CNN part
            return self.skip_fc5(x)
        else:  # otherwise we simply continue as in standard DQN and compute Q(s,a)
            x = F.relu(self.action_fc4(x.reshape(x.size(0), -1)))
            return self.action_fc5(x)

class B_HeadNet(nn.Module):
    """
    Largely a copy of JoungheeKim 
    See (https://github.com/JoungheeKim/bootsrapped-dqn)
    This code is for an ensemble network.

    """
    def __init__(self, reshape_size, num_actions=4):
        super(B_HeadNet, self).__init__()
        self.fc1 = nn.Linear(reshape_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class B_CoreNet(nn.Module):
    """
    Largely a copy of JoungheeKim 
    See (https://github.com/JoungheeKim/bootsrapped-dqn)
    This code is for an ensemble network.
    """
    def __init__(self, in_channels=4):
        super(B_CoreNet, self).__init__()
        # params from ddqn appendix
        if env.observation_space.shape[-1] == 84:  # hack
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        elif env.observation_space.shape[-1] == 42:  # hack
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2)
        else:
            raise ValueError("Check state space dimensionality. Expected nx42x42 or nx84x84. Was:",
                             env.observation_space.shape)
        # TODO - should we have this init during PRIOR code?
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.reshape_size = 64*7*7

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # size after conv3
        x = x.view(-1, self.reshape_size)
        return x

class B_EnsembleNet(nn.Module):
    """
    Largely a copy of JoungheeKim 
    See (https://github.com/JoungheeKim/bootsrapped-dqn)
    This code is for an ensemble network.
    """
    def __init__(self, in_channels=4, num_actions=18, n_heads=10):
        super(B_EnsembleNet, self).__init__()
        self.core_net = B_CoreNet(in_channels)
        reshape_size = self.core_net.reshape_size
        self.net_list = nn.ModuleList([B_HeadNet(reshape_size=reshape_size, num_actions=num_actions) for k in range(n_heads)])

    def _core(self, x):
        return self.core_net(x)

    def _heads(self, x):
        return [net(x) for net in self.net_list]

    def forward(self, x, k=None):
        if k is not None:
            return self.net_list[k](self.core_net(x))
        else:
            core_cache = self._core(x)
            net_heads = self._heads(core_cache)
            return net_heads

    def forward_mean(self, x):
        core_cache = self._core(x)
        net_heads = torch.stack(self._heads(core_cache))
        mean_Q = torch.mean(net_heads, axis=0)
        return mean_Q

class Extension_CoreNet(nn.Module):
    # Model style from Kyle @
    # https://gist.github.com/kastnerkyle/a4498fdf431a3a6d551bcc30cd9a35a0
    def __init__(self, in_channels=4):
        super(Extension_CoreNet, self).__init__()
        # params from ddqn appendix
        if env.observation_space.shape[-1] == 84:  # hack
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        elif env.observation_space.shape[-1] == 42:  # hack
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2)
        else:
            raise ValueError("Check state space dimensionality. Expected nx42x42 or nx84x84. Was:",
                             env.observation_space.shape)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.skip = nn.Linear(1, 10)  # Context layer
        self.apply(atari_initializer)

    def forward(self, x, action_val=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # size after conv3
        reshape = 64*7*7
        x = x.view(-1, reshape)
        if action_val is not None:
            x_ = F.relu(self.skip(action_val))
            x = torch.cat([x, x_], 1)
        return x

class NatureDQNhead(nn.Module):
    # Model style from Kyle @
    # https://gist.github.com/kastnerkyle/a4498fdf431a3a6d551bcc30cd9a35a0
    def __init__(self, core_net,  in_channels=4, num_actions=18):
        super(NatureDQNhead, self).__init__()
        self.core_net = core_net

        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)

        self.apply(atari_initializer)

    def forward(self, x):
        x = self.core_net(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Ensemble_Extension(nn.Module):
    def __init__(self, core_net, in_channels=4, num_actions=18, nheads=10):
        super(Ensemble_Extension, self).__init__()
        self.nheads = nheads
        self.core_net = core_net

        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(7 * 7 * 64 + 10, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, num_actions)) for _ in range(self.nheads)])

        self.apply(atari_initializer)

    def forward(self, x, action_val=None):
        torso = self.core_net(x, action_val)
        out = []
        for head in self.heads:
            out.append(head(torso))
        return out

    def forward_single_head(self, x, k, action_val=None):
        torso = self.core_net(x, action_val)
        out = self.heads[k](torso)
        return out

    def forward_mean(self, x, action_val=None):
        torso = self.core_net(x, action_val)
        out = []
        for head in self.heads:
            out.append(head(torso))
        out = torch.stack(out) # HxBx10
        out = torch.mean(out, axis=0) # Bx10
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
        return tt(batch_states), tt(batch_actions), tt(batch_next_states), tt(batch_rewards), tt(batch_terminal_flags)

    def save_buffer(self, filepath):
        np.savez_compressed(os.path.join(filepath, 'replay_buffer'),
                 states = self._data.states,
                 actions = self._data.actions,
                 next_states = self._data.next_states,
                 rewards = self._data.rewards,
                 terminal_flags = self._data.terminal_flags,
                 size = self._size,
                 max_size = self._max_size
                 )

    def load_buffer(self, filepath):
        npfile = np.load(os.path.join(filepath, 'replay_buffer.npz'), allow_pickle=True)
        states = npfile['states']
        actions = npfile['actions']
        next_states = npfile['next_states']
        rewards = npfile['rewards']
        terminal_flags = npfile['terminal_flags']
        self._size = npfile['size']
        self._max_size = npfile['max_size']
        npfile.close()

        for i in range(states.shape[0]):
            self._data.states.append(states[i])
            self._data.actions.append(actions[i])
            self._data.next_states.append(next_states[i])
            self._data.rewards.append(rewards[i])
            self._data.terminal_flags.append(terminal_flags[i])

class SkipReplayBuffer:
    """
    Replay Buffer for training the skip-Q.
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
        return tt(batch_states), tt(batch_actions), tt(batch_next_states),\
               tt(batch_rewards), tt(batch_terminal_flags), tt(batch_lengths)

    def save_buffer(self, filepath):
        np.savez_compressed(os.path.join(filepath, 'skip_replay_buffer'),
                 states = self._data.states,
                 actions = self._data.actions,
                 next_states = self._data.next_states,
                 rewards = self._data.rewards,
                 terminal_flags = self._data.terminal_flags,
                 lengths = self._data.lengths,
                 size = self._size,
                 max_size = self._max_size
                 )

    def load_buffer(self, filepath):
        npfile = np.load(os.path.join(filepath, 'skip_replay_buffer.npz'), allow_pickle=True)
        states = npfile['states']
        actions = npfile['actions']
        next_states = npfile['next_states']
        rewards = npfile['rewards']
        terminal_flags = npfile['terminal_flags']
        lengths = npfile['lengths']
        self._size = npfile['size']
        self._max_size = npfile['max_size']
        npfile.close()
        for i in range(states.shape[0]):
            self._data.states.append(states[i])
            self._data.actions.append(actions[i])
            self._data.next_states.append(next_states[i])
            self._data.rewards.append(rewards[i])
            self._data.terminal_flags.append(terminal_flags[i])  
            self._data.lengths.append(lengths[i])

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
        return tt(batch_states), tt(batch_actions), tt(batch_next_states),\
               tt(batch_rewards), tt(batch_terminal_flags), tt(batch_lengths), tt(batch_behavoiurs)
               
    def save_buffer(self, filepath):
        np.savez_compressed(os.path.join(filepath, 'skip_replay_buffer'),
                 states = self._data.states,
                 actions = self._data.actions,
                 next_states = self._data.next_states,
                 rewards = self._data.rewards,
                 terminal_flags = self._data.terminal_flags,
                 lengths = self._data.lengths,
                 behaviour_action = self._data.behaviour_action,
                 size = self._size,
                 max_size = self._max_size
                 )

    def load_buffer(self, filepath):
        npfile = np.load(os.path.join(filepath, 'skip_replay_buffer.npz'), allow_pickle=True)
        states = npfile['states']
        actions = npfile['actions']
        next_states = npfile['next_states']
        rewards = npfile['rewards']
        terminal_flags = npfile['terminal_flags']
        lengths = npfile['lengths']
        behaviour_action = npfile['behaviour_action']
        self._size = npfile['size']
        self._max_size = npfile['max_size']
        npfile.close()
        for i in range(states.shape[0]):
            self._data.states.append(states[i])
            self._data.actions.append(actions[i])
            self._data.next_states.append(next_states[i])
            self._data.rewards.append(rewards[i])
            self._data.terminal_flags.append(terminal_flags[i])  
            self._data.lengths.append(lengths[i])
            self._data.behaviour_action.append(behaviour_action[i])

class DQN:
    """
    Simple double DQN Agent
    """

    def __init__(self, state_dim: int, action_dim: int, gamma: float,
                 env: gym.Env, eval_env: gym.Env):
        """
        Initialize the DQN Agent
        :param state_dim: dimensionality of the input states
        :param action_dim: dimensionality of the output actions
        :param gamma: discount factor
        :param env: environment to train on
        :param eval_env: environment to evaluate on
        """
        self._q = NatureDQN(state_dim, action_dim).to(device)
        self._q_target = NatureDQN(state_dim, action_dim).to(device)

        self._gamma = gamma

        self.batch_size = 32
        self.grad_clip_val = 40.0
        self.target_net_upd_freq = 500
        self.learning_starts = 10_000
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.01
        self.epsilon_timesteps = 200_000
        self.train_freq = 4

        self._loss_function = nn.SmoothL1Loss()  # huber loss # nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
        self._action_dim = action_dim

        self._replay_buffer = ReplayBuffer(5e4)
        self._env = env
        self._eval_env = eval_env

        # Load model
        if args.load_dir:
            load_dir = os.path.join(args.out_dir, args.load_dir)   
            self.load_model(load_dir)
        else:
            self.total_steps = 0

    def get_action(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(tt(x[None, :])).cpu().detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def train(self, episodes: int, max_env_time_steps: int, epsilon: float, eval_eps: int = 1,
              eval_every_n_steps: int = 1, max_train_time_steps: int = 1_000_000):
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
        num_update_steps = 0
        batch_size = self.batch_size
        grad_clip_val = self.grad_clip_val
        target_net_upd_freq = self.target_net_upd_freq
        learning_starts = self.learning_starts

        start_time = time.time()

        for e in range(episodes):
            print("# Episode: %s/%s" % (e + 1, episodes))
            s = self._env.reset()

            for t in range(max_env_time_steps):
                if self.total_steps > self.epsilon_timesteps:
                    epsilon = self.final_epsilon
                else:
                    epsilon = self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * (
                                self.total_steps / self.epsilon_timesteps)

                a = self.get_action(s, epsilon)
                ns, r, d, _ = self._env.step(a)
                self.total_steps += 1

                ########### Begin Evaluation
                if (self.total_steps % eval_every_n_steps) == 0:
                    eval_s, eval_r, eval_d = self.eval(eval_eps, max_env_time_steps, epsilon=0.001)
                    eval_stats = dict(
                        elapsed_time=time.time() - start_time,
                        training_steps=self.total_steps,
                        training_eps=e,
                        avg_num_steps_per_eval_ep=float(np.round(np.mean(eval_s), 1)),
                        avg_num_decs_per_eval_ep=float(np.round(np.mean(eval_d), 1)),
                        avg_rew_per_eval_ep=float(np.round(np.mean(eval_r), 1)),
                        std_rew_per_eval_ep=float(np.round(np.std(eval_r), 1)),
                        eval_eps=eval_eps
                    )
                    print('Done %4d/%4d episodes, rewards: %4d' % (e, episodes, float(np.mean(eval_r))))
                    with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                        json.dump(eval_stats, out_fh)
                        out_fh.write('\n')     
                    self.save_model(out_dir)
                ########### End Evaluation

                # Update replay buffer
                self._replay_buffer.add_transition(s, a, ns, r, d)

                batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                    self._replay_buffer.random_next_batch(batch_size)

                ########### Begin double Q-learning update
                target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                         self._q_target(batch_next_states)[torch.arange(batch_size).long(), torch.argmax(
                             self._q(batch_next_states), dim=1)]
                current_prediction = self._q(batch_states)[torch.arange(batch_size).long(), batch_actions.long()]

                loss = self._loss_function(current_prediction, target.detach())

                if (self.total_steps > learning_starts) and (self.total_steps % self.train_freq == 0):
                    num_update_steps += 1
                    self._q_optimizer.zero_grad()
                    loss.backward()
                    for param in self._q.parameters():
                        param.grad.data.clamp_(-grad_clip_val, grad_clip_val)
                    self._q_optimizer.step()


                    if (self.total_steps % target_net_upd_freq) == 0:
                        hard_update(self._q_target, self._q)
                    # soft_update(self._q_target, self._q, 0.01)
                    ########### End double Q-learning update
                
                if d:
                    break
                s = ns
                if self.total_steps >= max_train_time_steps:
                    break
            if self.total_steps >= max_train_time_steps:
                break
            

    def eval(self, episodes: int, max_env_time_steps: int, epsilon:float):
        steps, rewards, decisions = [], [], []
        with torch.no_grad():
            for e in range(episodes):
                ed, es, er = 0, 0, 0

                s = self._eval_env.reset()
                for _ in count():
                    a = self.get_action(s, epsilon)
                    ed += 1

                    ns, r, d, _ = self._eval_env.step(a)
                    # print(r, d)
                    er += r
                    es += 1
                    if es >= max_env_time_steps or d:
                        break
                    s = ns
                steps.append(es)
                rewards.append(er)
                decisions.append(ed)

        return steps, rewards, decisions

    def save_model(self, path):
        torch.save(self._q.state_dict(), os.path.join(path, 'Q.pt'))
        torch.save({
            'total_steps': self.total_steps,
            'model_state_dict': self._q.state_dict(),
            'target_state_dict': self._q_target.state_dict(),
            'optimizer_state_dict': self._q_optimizer.state_dict(),
            }, os.path.join(path, 'Q.pt'))
        self._replay_buffer.save_buffer(path)

    def load_model(self, path):
        checkpoint = torch.load(os.path.join(path, 'Q.pt'))   
        self._q.load_state_dict(checkpoint['model_state_dict'])   
        self._q_target.load_state_dict(checkpoint['target_state_dict'])
        self._q_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self._replay_buffer.load_buffer(path)

class DAR:
    """
    Simple Dynamic Action Repetition Agent based on double DQN
    """

    def __init__(self, state_dim: int, action_dim: int,
                 num_output_duplication: int, skip_map: dict,
                 gamma: float, env: gym.Env, eval_env: gym.Env):
        """
        :param state_dim: dimensionality of the input states
        :param action_dim: dimensionality of the output actions
        :param num_output_duplication: integer that determines how often to duplicate output heads (original is 2)
        :param skip_map: determines the skip value associated with each output head
        :param gamma: discount factor
        :param env: environment to train on
        :param eval_env: environment to evaluate on
        """

        self._q = NatureDQN(state_dim, action_dim * num_output_duplication).to(device)
        self._q_target = NatureDQN(state_dim, action_dim * num_output_duplication).to(device)

        self._gamma = gamma

        self.batch_size = 32
        self.grad_clip_val = 40.0
        self.target_net_upd_freq = 500
        self.learning_starts = 10_000
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.01
        self.epsilon_timesteps = 200_000
        self.train_freq = 4   

        self._loss_function = nn.SmoothL1Loss()  # huber loss # nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
        self._action_dim = action_dim

        self._replay_buffer = ReplayBuffer(5e4)
        self._skip_map = skip_map
        self._dup_vals = num_output_duplication
        self._env = env
        self._eval_env = eval_env

        # Load model
        if args.load_dir:
            load_dir = os.path.join(args.out_dir, args.load_dir)   
            self.load_model(load_dir)
        else:
            self.total_steps = 0

    def get_action(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(tt(x[None, :])).detach().cpu().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim*self._dup_vals)
        return u

    def train(self, episodes: int, max_env_time_steps: int, epsilon: float, eval_eps: int = 1,
              eval_every_n_steps: int = 1, max_train_time_steps: int = 1_000_000):
        """
        Training loop
        :param episodes: maximum number of episodes to train for
        :param max_env_time_steps: maximum number of steps in the environment to perform per episode
        :param epsilon: constant epsilon for exploration when selecting actions
        :param eval_eps: numper of episodes to run for evaluation
        :param eval_every_n_steps: interval of steps after which to evaluate the trained agent
        :param max_train_time_steps: maximum number of steps to train
        """
        num_update_steps = 0
        batch_size = self.batch_size
        grad_clip_val = self.grad_clip_val
        target_net_upd_freq = self.target_net_upd_freq
        learning_starts = self.learning_starts

        start_time = time.time()
        for e in range(episodes):
            print("%s/%s" % (e + 1, episodes))
            s = self._env.reset()
            es = 0
            for t in range(max_env_time_steps):
                if self.total_steps > self.epsilon_timesteps:
                    epsilon = self.final_epsilon
                else:
                    epsilon = self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * (
                            self.total_steps / self.epsilon_timesteps)
                a = self.get_action(s, epsilon)

                # convert action id int corresponding behaviour action and skip value
                act = a // self._dup_vals # behaviour
                rep = a // self._env.action_space.n  # skip id
                skip = self._skip_map[rep]  # skip id to corresponding skip value

                for _ in range(skip + 1):  # repeat chosen behaviour action for "skip" steps
                    ns, r, d, _ = self._env.step(act)
                    self.total_steps += 1
                    es += 1

                    ########### Begin Evaluation
                    if (self.total_steps % eval_every_n_steps) == 0:
                        eval_s, eval_r, eval_d = self.eval(eval_eps, max_env_time_steps, epsilon=0.001)
                        eval_stats = dict(
                            elapsed_time=time.time() - start_time,
                            training_steps=self.total_steps,
                            training_eps=e,
                            avg_num_steps_per_eval_ep=float(np.round(np.mean(eval_s), 1)),
                            avg_num_decs_per_eval_ep=float(np.round(np.mean(eval_d), 1)),
                            avg_rew_per_eval_ep=float(np.round(np.mean(eval_r), 1)),
                            std_rew_per_eval_ep=float(np.round(np.std(eval_r), 1)),
                            eval_eps=eval_eps
                        )
                        print('Done %4d/%4d episodes, rewards: %4d' % (e, episodes, float(np.mean(eval_r))))
                        with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                            json.dump(eval_stats, out_fh)
                            out_fh.write('\n')
                            self.save_model(out_dir)
                    ########### End Evaluation
                    
                    ### Q-update based double Q learning
                    self._replay_buffer.add_transition(s, a , ns, r, d)

                    batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                        self._replay_buffer.random_next_batch(batch_size)

                    target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                             self._q_target(batch_next_states)[torch.arange(batch_size).long(), torch.argmax(
                                 self._q(batch_next_states), dim=1)]
                    current_prediction = self._q(batch_states)[torch.arange(batch_size).long(), batch_actions.long()]

                    loss = self._loss_function(current_prediction, target.detach())
                    if (self.total_steps > learning_starts) and (self.total_steps % self.train_freq == 0):
                        num_update_steps += 1
                        self._q_optimizer.zero_grad()
                        loss.backward()
                        for param in self._q.parameters():
                            param.grad.data.clamp_(-grad_clip_val, grad_clip_val)
                        self._q_optimizer.step()

                        if (self.total_steps % target_net_upd_freq) == 0:
                            hard_update(self._q_target, self._q)
                        # soft_update(self._q_target, self._q, 0.01)
                        ########### End double Q-learning update
                    
                    if es >= max_env_time_steps or d or self.total_steps >= max_train_time_steps:
                        break

                    s = ns
                if es >= max_env_time_steps or d or self.total_steps >= max_train_time_steps:
                    break

            if self.total_steps >= max_train_time_steps:
                break

    def eval(self, episodes: int, max_env_time_steps: int, epsilon:float):
        """
        Simple method that evaluates the agent with fixed epsilon = 0
        :param episodes: max number of episodes to play
        :param max_env_time_steps: max number of max_env_time_steps to play
        :returns (steps per episode), (reward per episode), (decisions per episode)
        """
        steps, rewards, decisions = [], [], []
        with torch.no_grad():
            for e in range(episodes):
                ed, es, er = 0, 0, 0

                s = self._eval_env.reset()
                for _ in count():
                    # print(self._q(tt(s)))
                    a = self.get_action(s, epsilon)
                    act = a % self._eval_env.action_space.n
                    rep = a // self._eval_env.action_space.n
                    skip = self._skip_map[rep]

                    ed += 1

                    d = False
                    for _ in range(skip + 1):
                        ns, r, d, _ = self._eval_env.step(act)
                        er += r
                        es += 1
                        if es >= max_env_time_steps or d:
                            break
                        s = ns
                    if es >= max_env_time_steps or d:
                        break
                steps.append(es)
                rewards.append(er)
                decisions.append(ed)

        return steps, rewards, decisions

    def save_model(self, path):
        torch.save(self._q.state_dict(), os.path.join(path, 'Q.pt'))
        torch.save({
            'total_steps': self.total_steps,
            'model_state_dict': self._q.state_dict(),
            'target_state_dict': self._q_target.state_dict(),
            'optimizer_state_dict': self._q_optimizer.state_dict(),
            }, os.path.join(path, 'Q.pt'))
        self._replay_buffer.save_buffer(path)

    def load_model(self, path):
        checkpoint = torch.load(os.path.join(path, 'Q.pt'))   
        self._q.load_state_dict(checkpoint['model_state_dict'])   
        self._q_target.load_state_dict(checkpoint['target_state_dict'])
        self._q_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self._replay_buffer.load_buffer(path)

class TDQN:
    """
    TempoRL DQN agent capable of handling more complex state inputs through use of contextualized behaviour actions.
    """

    def __init__(self, state_dim, action_dim, skip_dim, gamma, env, eval_env):
        """
        :param state_dim: dimensionality of the input states
        :param action_dim: dimensionality of the action output
        :param skip_dim: dimenionality of the skip output
        :param gamma: discount factor
        :param env: environment to train on
        :param eval_env: environment to evaluate on
        """
        self._q = NatureWeightsharingTQN(state_dim, action_dim, skip_dim).to(device)
        self._q_target = NatureWeightsharingTQN(state_dim, action_dim, skip_dim).to(device)
        self._skip_q = self._q

        print('Using {} as Q'.format(str(self._q)))
        print('Using {} as skip-Q\n{}'.format(str(self._skip_q), '#' * 80))

        self._gamma = gamma
        self._action_dim = action_dim
        self._skip_dim = skip_dim

        self.batch_size = 32
        self.grad_clip_val = 40.0
        self.target_net_upd_freq = 500
        self.learning_starts = 10_000
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.01
        self.epsilon_timesteps = 200_000
        self.train_freq = 4

        self._loss_function = nn.SmoothL1Loss()  # huber loss # nn.MSELoss()
        self._skip_loss_function = nn.SmoothL1Loss()  # nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
        self._skip_q_optimizer = optim.Adam(self._skip_q.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)

        self._replay_buffer = ReplayBuffer(5e4)
        self._skip_replay_buffer = NoneConcatSkipReplayBuffer(5e4)
        self._env = env
        self._eval_env = eval_env

        # Load model
        if args.load_dir:
            load_dir = os.path.join(args.out_dir, args.load_dir)   
            self.load_model(load_dir)
        else:
            self.total_steps = 0

    def get_action(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(tt(x[None, :])).cpu().detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def get_skip(self, x: np.ndarray, a: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get the skip epsilon-greedy based on observation x conditioned on behaviour action a
        """
        u = np.argmax(self._skip_q(tt(x[None, :]), tt(a[None, :])).detach().cpu().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._skip_dim)
        return u

    def train(self, episodes: int, max_env_time_steps: int, epsilon: float, eval_eps: int = 1,
              eval_every_n_steps: int = 1, max_train_time_steps: int = 1_000_000):
        """
        Training loop
        :param episodes: maximum number of episodes to train for
        :param max_env_time_steps: maximum number of steps in the environment to perform per episode
        :param epsilon: constant epsilon for exploration when selecting actions
        :param eval_eps: numper of episodes to run for evaluation
        :param eval_every_n_steps: interval of steps after which to evaluate the trained agent
        :param max_train_time_steps: maximum number of steps to train
        """
        num_update_steps = 0
        batch_size = self.batch_size
        grad_clip_val = self.grad_clip_val
        target_net_upd_freq = self.target_net_upd_freq
        learning_starts = self.learning_starts

        start_time = time.time()

        for e in range(episodes):
            print("# Episode: %s/%s" % (e + 1, episodes))
            s = self._env.reset()
            es = 0
            steps, rewards, decisions = [], [], []
            for _ in count():

                if self.total_steps > self.epsilon_timesteps:
                    epsilon = self.final_epsilon
                else:
                    epsilon = self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * (
                                self.total_steps / self.epsilon_timesteps)

                a = self.get_action(s, epsilon)
                skip = self.get_skip(s, np.array([a]), epsilon)  # get skip with the selected action as context

                d = False
                skip_states, skip_rewards = [], []
                for curr_skip in range(skip + 1):  # repeat the selected action for "skip" times
                    ns, r, d, info_ = self._env.step(a)
                    self.total_steps += 1
                    es += 1
                    skip_states.append(s)  # keep track of all observed skips
                    skip_rewards.append(r)

                    #### Begin Evaluation
                    if (self.total_steps % eval_every_n_steps) == 0:
                        eval_s, eval_r, eval_d = self.eval(eval_eps, max_env_time_steps, epsilon=0.001)
                        eval_stats = dict(
                            elapsed_time=time.time() - start_time,
                            training_steps=self.total_steps,
                            training_eps=e,
                            avg_num_steps_per_eval_ep=float(np.round(np.mean(eval_s), 1)),
                            avg_num_decs_per_eval_ep=float(np.round(np.mean(eval_d), 1)),
                            avg_rew_per_eval_ep=float(np.round(np.mean(eval_r), 1)),
                            std_rew_per_eval_ep=float(np.round(np.std(eval_r), 1)),
                            eval_eps=eval_eps
                        )
                        print('Done %4d/%4d episodes, rewards: %4d' % (e, episodes, float(np.mean(eval_r))))
                        with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                            json.dump(eval_stats, out_fh)
                            out_fh.write('\n')
                        self.save_model(out_dir)
                    #### End Evaluation

                    # Update the skip replay buffer with all observed skips.
                    skip_id = 0
                    for start_state in skip_states:
                        skip_reward = 0
                        for exp, r in enumerate(skip_rewards[skip_id:]):  # make sure to properly discount
                            skip_reward += np.power(self._gamma, exp) * r

                        self._skip_replay_buffer.add_transition(start_state, curr_skip - skip_id, ns,
                                                                skip_reward, d, curr_skip - skip_id + 1,
                                                                np.array([a]))  # also keep track of the behavior action
                        skip_id += 1

                    # Update the replay buffer
                    self._replay_buffer.add_transition(s, a, ns, r, d)

                    # Skip Q update based on double DQN where target is behavior Q
                    if (self.total_steps > learning_starts) and (self.total_steps % self.train_freq == 0):
                        num_update_steps += 1

                        batch_states, batch_actions, batch_next_states, batch_rewards,\
                            batch_terminal_flags, batch_lengths, batch_behaviours = \
                            self._skip_replay_buffer.random_next_batch(batch_size)

                        target = batch_rewards + (1 - batch_terminal_flags) * torch.pow(self._gamma, batch_lengths) * \
                                self._q_target(batch_next_states)[torch.arange(batch_size).long(), torch.argmax(
                                    self._q(batch_next_states), dim=1)]
                        current_prediction = self._skip_q(batch_states, batch_behaviours)[
                            torch.arange(batch_size).long(), batch_actions.long()]

                        loss = self._skip_loss_function(current_prediction, target.detach())

                        self._skip_q_optimizer.zero_grad()
                        loss.backward()
                        for param in self._skip_q.parameters():
                            if param.grad is None:
                                pass
                            else:
                                param.grad.data.clamp_(-grad_clip_val, grad_clip_val)
                        self._skip_q_optimizer.step()


                    # Action Q update based on double DQN with normal target
                    if (self.total_steps > learning_starts) and (self.total_steps % self.train_freq == 0):

                        batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                            self._replay_buffer.random_next_batch(batch_size)

                        target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                                self._q_target(batch_next_states)[torch.arange(batch_size).long(), torch.argmax(
                                    self._q(batch_next_states), dim=1)]
                        current_prediction = self._q(batch_states)[torch.arange(batch_size).long(), batch_actions.long()]

                        loss = self._loss_function(current_prediction, target.detach())

                        self._q_optimizer.zero_grad()
                        loss.backward()
                        for param in self._q.parameters():
                            if param.grad is None:
                                pass
                                # print("##### Q Parameter with grad = None:", param.name)
                            else:
                                param.grad.data.clamp_(-grad_clip_val, grad_clip_val)
                        self._q_optimizer.step()

                        if (self.total_steps % target_net_upd_freq) == 0:
                            hard_update(self._q_target, self._q)
                        # soft_update(self._q_target, self._q, 0.01)

                    if es >= max_env_time_steps or d or self.total_steps >= max_train_time_steps:
                        break
                    s = ns
                if es >= max_env_time_steps or d or self.total_steps >= max_train_time_steps:
                    break
            if self.total_steps >= max_train_time_steps:
                break

    def eval(self, episodes: int, max_env_time_steps: int, epsilon: float):
        """
        Simple method that evaluates the agent
        :param episodes: max number of episodes to play
        :param max_env_time_steps: max number of max_env_time_steps to play
        :returns (steps per episode), (reward per episode), (decisions per episode)
        """
        steps, rewards, decisions = [], [], []
        with torch.no_grad():
            for e in range(episodes):
                ed, es, er = 0, 0, 0

                s = self._eval_env.reset()
                for _ in count():
                    a = self.get_action(s, epsilon)
                    skip = self.get_skip(s, np.array([a]), epsilon)
                    ed += 1

                    d = False
                    for _ in range(skip + 1):
                        ns, r, d, _ = self._eval_env.step(a)
                        er += r
                        es += 1
                        if es >= max_env_time_steps or d:
                            break
                        s = ns
                    if es >= max_env_time_steps or d:
                        break
                steps.append(es)
                rewards.append(er)
                decisions.append(ed)

        return steps, rewards, decisions

    def save_model(self, path):
        torch.save({
            'total_steps': self.total_steps,
            'model_state_dict': self._q.state_dict(),
            'target_state_dict': self._q_target.state_dict(),
            'skip_q_state_dict': self._skip_q.state_dict(),
            'optimizer_state_dict': self._q_optimizer.state_dict(),
            'skip_optimizer_state_dict' : self._skip_q_optimizer.state_dict()
            }, os.path.join(path, 'Q.pt'))
        self._replay_buffer.save_buffer(path)
        self._skip_replay_buffer.save_buffer(path)

    def load_model(self, path):
        checkpoint = torch.load(os.path.join(path, 'Q.pt'))   
        self._q.load_state_dict(checkpoint['model_state_dict'])   
        self._q_target.load_state_dict(checkpoint['target_state_dict'])
        self._q_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._skip_q.load_state_dict(checkpoint['skip_q_state_dict'])
        self._skip_q_optimizer.load_state_dict(checkpoint['skip_optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self._replay_buffer.load_buffer(path)
        self._skip_replay_buffer.load_buffer(path)

class UTE_One_Step:
    """
    UTE without n-step Q learning
    """

    def __init__(self, state_dim, action_dim, skip_dim, gamma, env, eval_env, uncertainty_factor):
        """
        :param state_dim: dimensionality of the input states
        :param action_dim: dimensionality of the action output
        :param skip_dim: dimenionality of the skip output
        :param gamma: discount factor
        :param env: environment to train on
        :param eval_env: environment to evaluate on
        """
        self.core_net = Extension_CoreNet()
        
        self.core_net_target = Extension_CoreNet()
        self._q = NatureDQNhead(self.core_net, state_dim, action_dim).to(device)
        self._q_target = NatureDQNhead(self.core_net_target, state_dim, action_dim).to(device)       

        self.n_heads = 10
        self._skip_q = Ensemble_Extension(self.core_net, state_dim, skip_dim, self.n_heads).to(device)

        print('Using {} as Q'.format(str(self._q)))
        print('Using {} as skip-Q\n{}'.format(str(self._skip_q), '#' * 80))

        self._gamma = gamma
        self._action_dim = action_dim
        self._skip_dim = skip_dim

        self.batch_size = 32
        self.grad_clip_val = 40.0
        self.target_net_upd_freq = 500
        self.learning_starts = 10_000
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.01
        self.epsilon_timesteps = 200_000
        self.train_freq = 4
        self.bernoulli_probability = 0.5
        self.uncertainty_factor = uncertainty_factor

        self._time_limt = 42500

        self._loss_function = nn.SmoothL1Loss()  # huber loss # nn.MSELoss()
        self._skip_loss_function = nn.SmoothL1Loss()  # nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
        self._skip_q_optimizer = optim.Adam(self._skip_q.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
        
        self._replay_buffer = ReplayBuffer(5e4)
        self._skip_replay_buffer = NoneConcatSkipReplayBuffer(5e4)
        self._env = env
        self._eval_env = eval_env

        # Load model
        if args.load_dir:
            load_dir = os.path.join(args.out_dir, args.load_dir)   
            self.load_model(load_dir)
        else:
            self.total_steps = 0

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
        current_outputs = self._skip_q(tt(x[None, :]), tt(a[None, :]))
        outputs = []
        for k in range(self.n_heads):
            outputs.append(current_outputs[k].detach().cpu().numpy())
        outputs = np.array(outputs)
        mean_Q = np.mean(outputs, axis=0) 
        std_Q = np.std(outputs, axis=0)
        Q_tilda = mean_Q + self.uncertainty_factor*std_Q
        if Q_tilda.shape[0] > 1 :
            u = np.argmax(Q_tilda, axis=-1)
        else:
            u = np.argmax(Q_tilda)
        return u

    def train(self, episodes: int, max_env_time_steps: int, epsilon: float, eval_eps: int = 1,
              eval_every_n_steps: int = 1, max_train_time_steps: int = 1_000_000):
        """
        Training loop
        :param episodes: maximum number of episodes to train for
        :param max_env_time_steps: maximum number of steps in the environment to perform per episode
        :param epsilon: constant epsilon for exploration when selecting actions
        :param eval_eps: numper of episodes to run for evaluation
        :param eval_every_n_steps: interval of steps after which to evaluate the trained agent
        :param max_train_time_steps: maximum number of steps to train
        """
        num_update_steps = 0
        batch_size = self.batch_size
        grad_clip_val = self.grad_clip_val
        target_net_upd_freq = self.target_net_upd_freq
        learning_starts = self.learning_starts

        start_time = time.time()

        for e in range(episodes):
            print("# Episode: %s/%s" % (e + 1, episodes))
            s = self._env.reset()
            es = 0
            for _ in count():

                if self.total_steps > self.epsilon_timesteps:
                    epsilon = self.final_epsilon
                else:
                    epsilon = self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * (
                                self.total_steps / self.epsilon_timesteps)

                a = self.get_action(s, epsilon)
                skip = self.get_skip(s, np.array([a]))  # get skip with the selected action as context

                d = False
                skip_states, skip_rewards = [], []
                for curr_skip in range(skip + 1):  # repeat the selected action for "skip" times
                    ns, r, d, info_ = self._env.step(a)
                    self.total_steps += 1
                    es += 1
                    skip_states.append(s)  # keep track of all observed skips
                    skip_rewards.append(r)

                    #### Begin Evaluation
                    if (self.total_steps % eval_every_n_steps) == 0:
                        eval_s, eval_r, eval_d = self.eval(eval_eps, max_env_time_steps, epsilon=0.001)
                        eval_stats = dict(
                            elapsed_time=time.time() - start_time,
                            training_steps=self.total_steps,
                            training_eps=e,
                            avg_num_steps_per_eval_ep=float(np.round(np.mean(eval_s), 1)),
                            avg_num_decs_per_eval_ep=float(np.round(np.mean(eval_d), 1)),
                            avg_rew_per_eval_ep=float(np.round(np.mean(eval_r), 1)),
                            std_rew_per_eval_ep=float(np.round(np.std(eval_r), 1)),
                            eval_eps=eval_eps
                        )
                        print('Done %4d/%4d episodes, rewards: %4d' % (e, episodes, float(np.mean(eval_r))))
                        with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                            json.dump(eval_stats, out_fh)
                            out_fh.write('\n')
                        self.save_model(out_dir)
                    #### End Evaluation

                    # Update the skip replay buffer with all observed skips.
                    skip_id = 0
                    for start_state in skip_states:
                        skip_reward = 0
                        for exp, r in enumerate(skip_rewards[skip_id:]):  # make sure to properly discount
                            skip_reward += np.power(self._gamma, exp) * r

                        self._skip_replay_buffer.add_transition(start_state, curr_skip - skip_id, ns,
                                                                skip_reward, d, curr_skip - skip_id + 1,
                                                                np.array([a]))  # also keep track of the behavior action
                        skip_id += 1

                    # Update the replay buffer
                    self._replay_buffer.add_transition(s, a, ns, r, d)

                    # Skip Q update based on double DQN where target is behavior Q
                    if (self.total_steps > learning_starts) and (self.total_steps % self.train_freq == 0):
                        num_update_steps += 1

                        batch_states, batch_actions, batch_next_states, batch_rewards,\
                            batch_terminal_flags, batch_lengths, batch_behaviours = \
                            self._skip_replay_buffer.random_next_batch(batch_size*2)

                        target = batch_rewards + (1 - batch_terminal_flags) * torch.pow(self._gamma, batch_lengths) * \
                                self._q_target(batch_next_states)[torch.arange(batch_size*2).long(), torch.argmax(
                                    self._q(batch_next_states), dim=1)]
                        
                        current_outputs = self._skip_q(batch_states, batch_behaviours)
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
                        for param in self._skip_q.core_net.parameters():
                            if param.grad is None:
                                pass
                            else:
                                param.grad.data *= 1.0/float(self.n_heads)
                        for param in self._skip_q.parameters():
                            if param.grad is None:
                                pass
                            else:
                                param.grad.data.clamp_(-grad_clip_val, grad_clip_val)
                        self._skip_q_optimizer.step()

                    
                    # Action Q update based on double DQN with normal target
                    if (self.total_steps > learning_starts) and (self.total_steps % self.train_freq == 0):

                        batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                            self._replay_buffer.random_next_batch(batch_size)

                        target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                                self._q_target(batch_next_states)[torch.arange(batch_size).long(), torch.argmax(
                                    self._q(batch_next_states), dim=1)]
                        current_prediction = self._q(batch_states)[torch.arange(batch_size).long(), batch_actions.long()]

                        loss = self._loss_function(current_prediction, target.detach())

                        self._q_optimizer.zero_grad()
                        loss.backward()
                        for param in self._q.parameters():
                            if param.grad is None:
                                pass
                                # print("##### Q Parameter with grad = None:", param.name)
                            else:
                                param.grad.data.clamp_(-grad_clip_val, grad_clip_val)
                        self._q_optimizer.step()

                        if (self.total_steps % target_net_upd_freq) == 0:
                            hard_update(self._q_target, self._q)
                        # soft_update(self._q_target, self._q, 0.01)

                    if es >= max_env_time_steps or d or self.total_steps >= max_train_time_steps:
                        break
                    s = ns
                if es >= max_env_time_steps or d or self.total_steps >= max_train_time_steps or ((time.time() - start_time )>self._time_limt):
                    break
            if self.total_steps >= max_train_time_steps or ((time.time() - start_time )>self._time_limt):
                break

    def eval(self, episodes: int, max_env_time_steps: int, epsilon: float):
        """
        Simple method that evaluates the agent
        :param episodes: max number of episodes to play
        :param max_env_time_steps: max number of max_env_time_steps to play
        :returns (steps per episode), (reward per episode), (decisions per episode)
        """
        steps, rewards, decisions = [], [], []
        with torch.no_grad():
            for e in range(episodes):
                ed, es, er = 0, 0, 0

                s = self._eval_env.reset()
                for _ in count():
                    a = self.get_action(s, epsilon)
                    skip = self.get_skip(s, np.array([a]))
                    ed += 1

                    d = False
                    for _ in range(skip + 1):
                        ns, r, d, _ = self._eval_env.step(a)
                        er += r
                        es += 1
                        if es >= max_env_time_steps or d:
                            break
                        s = ns
                    if es >= max_env_time_steps or d:
                        break
                steps.append(es)
                rewards.append(er)
                decisions.append(ed)

        return steps, rewards, decisions

    def save_model(self, path):
        torch.save({
            'total_steps': self.total_steps,
            'model_state_dict': self._q.state_dict(),
            'target_state_dict': self._q_target.state_dict(),
            'skip_q_state_dict': self._skip_q.state_dict(),
            'optimizer_state_dict': self._q_optimizer.state_dict(),
            'skip_optimizer_state_dict' : self._skip_q_optimizer.state_dict()
            }, os.path.join(path, 'Q.pt'))
        self._replay_buffer.save_buffer(path)
        self._skip_replay_buffer.save_buffer(path)

    def load_model(self, path):
        checkpoint = torch.load(os.path.join(path, 'Q.pt'))   
        self._q.load_state_dict(checkpoint['model_state_dict'])   
        self._q_target.load_state_dict(checkpoint['target_state_dict'])
        self._q_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._skip_q.load_state_dict(checkpoint['skip_q_state_dict'])
        self._skip_q_optimizer.load_state_dict(checkpoint['skip_optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self._replay_buffer.load_buffer(path)
        self._skip_replay_buffer.load_buffer(path)

class UTE:
    """
    UTE: Uncertainty-aware Temporal Extension
    """

    def __init__(self, state_dim, action_dim, skip_dim, gamma, env, eval_env, uncertainty_factor):

        self.core_net = Extension_CoreNet()
        self.core_net_target = Extension_CoreNet()
        self._q = NatureDQNhead(self.core_net, state_dim, action_dim).to(device)
        self._q_target = NatureDQNhead(self.core_net_target, state_dim, action_dim).to(device)       

        self.n_heads = 10
        self._skip_q = Ensemble_Extension(self.core_net, state_dim, skip_dim, self.n_heads).to(device)
        print('Using {} as Q'.format(str(self._q)))
        print('Using {} as skip-Q\n{}'.format(str(self._skip_q), '#' * 80))

        self._gamma = gamma
        self._action_dim = action_dim
        self._skip_dim = skip_dim

        self.batch_size = 32
        self.grad_clip_val = 40.0
        self.target_net_upd_freq = 500
        self.learning_starts = 10_000
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.01
        self.epsilon_timesteps = 200_000
        self.train_freq = 4
        self.bernoulli_probability = 0.5
        self.uncertainty_factor = uncertainty_factor 

        self._loss_function = nn.SmoothL1Loss()  # huber loss # nn.MSELoss()
        self._skip_loss_function = nn.SmoothL1Loss()  # nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
        self._skip_q_optimizer = optim.Adam(self._skip_q.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
        
        self._skip_replay_buffer = NoneConcatSkipReplayBuffer(1e5)
        self._env = env
        self._eval_env = eval_env

        self._time_limt = 42500

        # Load model
        if args.load_dir:
            load_dir = os.path.join(args.out_dir, args.load_dir)   
            self.load_model(load_dir)
        else:
            self.total_steps = 0

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
        current_outputs = self._skip_q(tt(x[None, :]), tt(a[None, :]))
        outputs = []
        for k in range(self.n_heads):
            outputs.append(current_outputs[k].detach().cpu().numpy())
        outputs = np.array(outputs)
        mean_Q = np.mean(outputs , axis=0) # 1x10
        std_Q = np.std(outputs, axis=0)
        Q_tilda = mean_Q + self.uncertainty_factor*std_Q
        u = np.argmax(Q_tilda)
        return u

    def get_batch_skip(self, x , a):
        current_outputs = self._skip_q(x, a)
        outputs = []
        for k in range(self.n_heads):
            outputs.append(current_outputs[k].detach().cpu().numpy())  # Bx10
        outputs = np.array(outputs) # HxBx10
        mean_Q = np.mean(outputs, axis=0) # Bx10
        std_Q = np.std(outputs, axis=0)
        Q_tilda = mean_Q + self.uncertainty_factor*std_Q
        u = np.argmax(Q_tilda, axis=-1) #B
        return u

    def train(self, episodes: int, max_env_time_steps: int, epsilon: float, eval_eps: int = 1,
              eval_every_n_steps: int = 1, max_train_time_steps: int = 1_000_000):
        """
        Training loop
        :param episodes: maximum number of episodes to train for
        :param max_env_time_steps: maximum number of steps in the environment to perform per episode
        :param epsilon: constant epsilon for exploration when selecting actions
        :param eval_eps: numper of episodes to run for evaluation
        :param eval_every_n_steps: interval of steps after which to evaluate the trained agent
        :param max_train_time_steps: maximum number of steps to train
        """
        num_update_steps = 0
        batch_size = self.batch_size
        grad_clip_val = self.grad_clip_val
        target_net_upd_freq = self.target_net_upd_freq
        learning_starts = self.learning_starts

        start_time = time.time()

        for e in range(episodes):
            print("# Episode: %s/%s" % (e + 1, episodes))
            s = self._env.reset()
            es = 0
            for _ in count():

                if self.total_steps > self.epsilon_timesteps:
                    epsilon = self.final_epsilon
                else:
                    epsilon = self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * (
                                self.total_steps / self.epsilon_timesteps)

                a = self.get_action(s, epsilon)
                skip = self.get_skip(s, np.array([a]))  # get skip with the selected action as context

                d = False
                skip_states, skip_rewards = [], []
                for curr_skip in range(skip + 1):  # repeat the selected action for "skip" times
                    ns, r, d, info_ = self._env.step(a)
                    self.total_steps += 1
                    es += 1
                    skip_states.append(s)  # keep track of all observed skips
                    skip_rewards.append(r)

                    #### Begin Evaluation
                    if (self.total_steps % eval_every_n_steps) == 0:
                        eval_s, eval_r, eval_d = self.eval(eval_eps, max_env_time_steps, epsilon=0.001)
                        eval_stats = dict(
                            elapsed_time=time.time() - start_time,
                            training_steps=self.total_steps,
                            training_eps=e,
                            avg_num_steps_per_eval_ep=float(np.round(np.mean(eval_s), 1)),
                            avg_num_decs_per_eval_ep=float(np.round(np.mean(eval_d), 1)),
                            avg_rew_per_eval_ep=float(np.round(np.mean(eval_r), 1)),
                            std_rew_per_eval_ep=float(np.round(np.std(eval_r), 1)),
                            eval_eps=eval_eps
                        )
                        print('Done %4d/%4d episodes, rewards: %4d' % (e, episodes, float(np.mean(eval_r))))
                        with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                            json.dump(eval_stats, out_fh)
                            out_fh.write('\n')
                        self.save_model(out_dir)
                    #### End Evaluation

                    # Update the skip replay buffer with all observed skips.
                    skip_id = 0
                    for start_state in skip_states:
                        skip_reward = 0
                        for exp, r in enumerate(skip_rewards[skip_id:]):  # make sure to properly discount
                            skip_reward += np.power(self._gamma, exp) * r

                        self._skip_replay_buffer.add_transition(start_state, curr_skip - skip_id, ns,
                                                                skip_reward, d, curr_skip - skip_id + 1,
                                                                np.array([a]))  # also keep track of the behavior action
                        skip_id += 1

                    # Skip Q update based on double DQN where target is behavior Q
                    if (self.total_steps > learning_starts) and (self.total_steps % self.train_freq == 0):
                        num_update_steps += 1

                        batch_states, batch_actions, batch_next_states, batch_rewards,\
                            batch_terminal_flags, batch_lengths, batch_behaviours = \
                            self._skip_replay_buffer.random_next_batch(batch_size*2)

                        target = batch_rewards + (1 - batch_terminal_flags) * torch.pow(self._gamma, batch_lengths) * \
                                self._q_target(batch_next_states)[torch.arange(batch_size*2).long(), torch.argmax(
                                    self._q(batch_next_states), dim=1)]
                        
                        current_outputs = self._skip_q(batch_states, batch_behaviours)
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
                        for param in self._skip_q.core_net.parameters():
                            if param.grad is None:
                                pass
                            else:
                                param.grad.data *= 1.0/float(self.n_heads)
                        for param in self._skip_q.parameters():
                            if param.grad is None:
                                pass
                            else:
                                param.grad.data.clamp_(-grad_clip_val, grad_clip_val)
                        self._skip_q_optimizer.step()

                    
                    # Action Q update based on double DQN with nstep target
                    if (self.total_steps > learning_starts) and (self.total_steps % self.train_freq == 0):

                        batch_states, batch_skips, batch_next_states, batch_rewards,\
                            batch_terminal_flags, batch_lengths, batch_behaviours = \
                            self._skip_replay_buffer.random_next_batch(batch_size*3)

                        current_skip_prediction = self.get_batch_skip(batch_states, batch_behaviours) 
                        batch_use = np.where(batch_skips.squeeze().detach().cpu().numpy() <= current_skip_prediction)[0]
                        tmp_batch_size = len(batch_use)
                        if tmp_batch_size > batch_size:
                            batch_use = batch_use[:batch_size]
                            tmp_batch_size = len(batch_use)

                        batch_states = batch_states[batch_use]
                        batch_next_states = batch_next_states[batch_use]
                        batch_rewards = batch_rewards[batch_use]
                        batch_terminal_flags = batch_terminal_flags[batch_use]
                        batch_lengths = batch_lengths[batch_use]
                        batch_behaviours = batch_behaviours[batch_use]

                        next_actions = torch.argmax(self._q(batch_next_states), dim=1)
                        target = batch_rewards + (1 - batch_terminal_flags) * torch.pow(self._gamma, batch_lengths)* \
                                self._q_target(batch_next_states)[torch.arange(tmp_batch_size).long(), next_actions]
                        current_prediction = self._q(batch_states)[torch.arange(tmp_batch_size).long(), batch_behaviours.squeeze().long()]
                        loss = self._loss_function(current_prediction, target.detach())

                        self._q_optimizer.zero_grad()
                        loss.backward()
                        for param in self._q.parameters():
                            if param.grad is None:
                                pass
                            else:
                                param.grad.data.clamp_(-grad_clip_val, grad_clip_val)
                        self._q_optimizer.step()

                        if (self.total_steps % target_net_upd_freq) == 0:
                            hard_update(self._q_target, self._q)
                        # soft_update(self._q_target, self._q, 0.01)

                    if es >= max_env_time_steps or d or self.total_steps >= max_train_time_steps or ((time.time() - start_time )>self._time_limt):
                        break
                    s = ns
                if es >= max_env_time_steps or d or self.total_steps >= max_train_time_steps or ((time.time() - start_time )>self._time_limt):
                    break
            if self.total_steps >= max_train_time_steps or ((time.time() - start_time )>self._time_limt):
                break

    def eval(self, episodes: int, max_env_time_steps: int, epsilon: float):
        """
        Simple method that evaluates the agent
        :param episodes: max number of episodes to play
        :param max_env_time_steps: max number of max_env_time_steps to play
        :returns (steps per episode), (reward per episode), (decisions per episode)
        """
        steps, rewards, decisions = [], [], []
        with torch.no_grad():
            for e in range(episodes):
                ed, es, er = 0, 0, 0

                s = self._eval_env.reset()
                for _ in count():
                    a = self.get_action(s, epsilon)
                    skip = self.get_skip(s, np.array([a]))
                    ed += 1
                    d = False
                    for _ in range(skip + 1):
                        ns, r, d, _ = self._eval_env.step(a)
                        er += r
                        es += 1
                        if es >= max_env_time_steps or d:
                            break
                        s = ns
                    if es >= max_env_time_steps or d:
                        break
                steps.append(es)
                rewards.append(er)
                decisions.append(ed)

        return steps, rewards, decisions

    def save_model(self, path):
        torch.save({
            'total_steps': self.total_steps,
            'model_state_dict': self._q.state_dict(),
            'target_state_dict': self._q_target.state_dict(),
            'skip_q_state_dict': self._skip_q.state_dict(),
            'optimizer_state_dict': self._q_optimizer.state_dict(),
            'skip_optimizer_state_dict' : self._skip_q_optimizer.state_dict()
            }, os.path.join(path, 'Q.pt'))
        self._skip_replay_buffer.save_buffer(path)

    def load_model(self, path):
        checkpoint = torch.load(os.path.join(path, 'Q.pt'))   
        self._q.load_state_dict(checkpoint['model_state_dict'])   
        self._q_target.load_state_dict(checkpoint['target_state_dict'])
        self._q_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._skip_q.load_state_dict(checkpoint['skip_q_state_dict'])
        self._skip_q_optimizer.load_state_dict(checkpoint['skip_optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self._skip_replay_buffer.load_buffer(path)

class BootstrappedDQN:
    """
    Based Bootstrapped DQN paper, Osband et al. 2016.
    """
    def __init__(self, state_dim: int, action_dim: int, gamma: float,
                 env: gym.Env, eval_env: gym.Env):
        """
        :param state_dim: dimensionality of the input states
        :param action_dim: dimensionality of the output actions
        :param gamma: discount factor
        :param env: environment to train on
        :param eval_env: environment to evaluate on
        :param vision: boolean flag to indicate if the input state is an image or not
        """
        self.n_heads = 10
        self._q = B_EnsembleNet(state_dim, action_dim, self.n_heads).to(device)
        self._q_target = B_EnsembleNet(state_dim, action_dim, self.n_heads).to(device)

        self._gamma = gamma

        self.batch_size = 32
        self.grad_clip_val = 40.0
        self.target_net_upd_freq = 500
        self.learning_starts = 10_000
        self.epsilon_timesteps = 200_000
        self.train_freq = 4

        self._loss_function = nn.SmoothL1Loss()  # huber loss # nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
        self._action_dim = action_dim

        self._replay_buffer = ReplayBuffer(5e4)  
        self._env = env
        self._eval_env = eval_env

        self.bernoulli_probability = 0.9
        self._time_limt = 42500

        # Load model
        if args.load_dir:
            load_dir = os.path.join(args.out_dir, args.load_dir)   
            self.load_model(load_dir)
        else:
            self.total_steps = 0

    def get_action(self, x: np.ndarray, header_number:int=None) -> int:
        with torch.no_grad():
            if header_number is not None:
                action = self._q(tt(x[None, :]),header_number).cpu()
                return int(action.max(1).indices.numpy())
            else:
                # vote
                actions = self._q(tt(x[None, :]))
                actions = [int(action.cpu().max(1).indices.numpy()) for action in actions]
                actions = Counter(actions)
                action = actions.most_common(1)[0][0]
                return action                

    def train(self, episodes: int, max_env_time_steps: int, epsilon: float, eval_eps: int = 1,
              eval_every_n_steps: int = 1, max_train_time_steps: int = 1_000_000):
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
        num_update_steps = 0
        batch_size = self.batch_size
        grad_clip_val = self.grad_clip_val
        target_net_upd_freq = self.target_net_upd_freq
        learning_starts = self.learning_starts

        start_time = time.time()

        for e in range(episodes):
            print("# Episode: %s/%s" % (e + 1, episodes))
            s = self._env.reset()

            # Sample Active heads
            heads = list(range(self.n_heads))
            np.random.shuffle(heads)
            active_head = heads[0]            

            for t in range(max_env_time_steps):

                a = self.get_action(s, active_head)
                ns, r, d, _ = self._env.step(a)
                self.total_steps += 1

                ########### Begin Evaluation
                if (self.total_steps % eval_every_n_steps) == 0:
                    eval_s, eval_r, eval_d = self.eval(eval_eps, max_env_time_steps)
                    eval_stats = dict(
                        elapsed_time=time.time() - start_time,
                        training_steps=self.total_steps,
                        training_eps=e,
                        avg_num_steps_per_eval_ep=float(np.round(np.mean(eval_s), 1)),
                        avg_num_decs_per_eval_ep=float(np.round(np.mean(eval_d), 1)),
                        avg_rew_per_eval_ep=float(np.round(np.mean(eval_r), 1)),
                        std_rew_per_eval_ep=float(np.round(np.std(eval_r), 1)),
                        eval_eps=eval_eps
                    )
                    print('Done %4d/%4d episodes, rewards: %4d' % (e, episodes, float(np.mean(eval_r))))
                    with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                        json.dump(eval_stats, out_fh)
                        out_fh.write('\n')     
                    self.save_model(out_dir)
                ########### End Evaluation

                # Update replay buffer
                self._replay_buffer.add_transition(s, a, ns, r, d)

                if (self.total_steps > learning_starts) and (self.total_steps % self.train_freq == 0):
                    batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                        self._replay_buffer.random_next_batch(batch_size)

                    ########### Begin double Q-learning update
                    next_q_target = self._q_target(batch_next_states)
                    next_q = self._q(batch_next_states)
                    current_q = self._q(batch_states)
                    masks = torch.bernoulli(torch.zeros((batch_size, self.n_heads), device=device) + self.bernoulli_probability )
                    cnt_losses = []
                    for k in range(self.n_heads):
                        total_used = torch.sum(masks[:,k])
                        if total_used > 0.0:
                            target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                                    next_q_target[k][torch.arange(batch_size).long(), torch.argmax(
                                        next_q[k], dim=1)]

                            current_prediction = current_q[k][torch.arange(batch_size).long(), batch_actions.long()]
                            l1loss = self._loss_function(current_prediction, target.detach())
                            full_loss = masks[:,k]*l1loss
                            loss = torch.sum(full_loss/total_used)
                            cnt_losses.append(loss)

                    num_update_steps += 1
                    self._q_optimizer.zero_grad()
                    loss = sum(cnt_losses)/self.n_heads
                    loss.backward()
                    for param in self._q.core_net.parameters():
                        if param.grad is None:
                            pass
                        else:
                            param.grad.data *= 1.0/float(self.n_heads)
                    for param in self._q.parameters():
                        param.grad.data.clamp_(-grad_clip_val, grad_clip_val)
                    self._q_optimizer.step()


                    if (self.total_steps % target_net_upd_freq) == 0:
                        hard_update(self._q_target, self._q)
                    # soft_update(self._q_target, self._q, 0.01)
                    ########### End double Q-learning update
                
                if d:
                    break
                s = ns
                if self.total_steps >= max_train_time_steps or ((time.time() - start_time )>self._time_limt):
                    break
            if self.total_steps >= max_train_time_steps or ((time.time() - start_time )>self._time_limt):
                break
            

    def eval(self, episodes: int, max_env_time_steps: int):
        steps, rewards, decisions = [], [], []
        with torch.no_grad():
            for e in range(episodes):
                ed, es, er = 0, 0, 0

                s = self._eval_env.reset()
                for _ in count():
                    a = self.get_action(s)
                    ed += 1
                    ns, r, d, _ = self._eval_env.step(a)
                    er += r
                    es += 1
                    if es >= max_env_time_steps or d:
                        break
                    s = ns
                steps.append(es)
                rewards.append(er)
                decisions.append(ed)

        return steps, rewards, decisions

    def save_model(self, path):
        torch.save(self._q.state_dict(), os.path.join(path, 'Q.pt'))
        torch.save({
            'total_steps': self.total_steps,
            'model_state_dict': self._q.state_dict(),
            'target_state_dict': self._q_target.state_dict(),
            'optimizer_state_dict': self._q_optimizer.state_dict(),
            }, os.path.join(path, 'Q.pt'))
        self._replay_buffer.save_buffer(path)

    def load_model(self, path):
        checkpoint = torch.load(os.path.join(path, 'Q.pt'))   
        self._q.load_state_dict(checkpoint['model_state_dict'])   
        self._q_target.load_state_dict(checkpoint['target_state_dict'])
        self._q_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self._replay_buffer.load_buffer(path)

class Full_Ensemble:
    def __init__(self, state_dim, action_dim, skip_dim, gamma, env, eval_env, uncertainty_factor):
        """
        Initialize the Agent
        :param state_dim: dimensionality of the input states
        :param action_dim: dimensionality of the output actions
        :param gamma: discount factor
        :param env: environment to train on
        :param eval_env: environment to evaluate on
        :param vision: boolean flag to indicate if the input state is an image or not
        """
        self.core_net = Extension_CoreNet()
        self.n_heads = 10
        self._q = B_EnsembleNet(state_dim, action_dim, self.n_heads).to(device)
        self._q_target = B_EnsembleNet(state_dim, action_dim, self.n_heads).to(device)

        self._skip_q = Ensemble_Extension(self.core_net, state_dim, skip_dim, self.n_heads).to(device)

        print('Using {} as Q'.format(str(self._q)))
        print('Using {} as skip-Q\n{}'.format(str(self._skip_q), '#' * 80))

        self._gamma = gamma
        self._action_dim = action_dim
        self._skip_dim = skip_dim

        self.batch_size = 32
        self.grad_clip_val = 40.0
        self.target_net_upd_freq = 500
        self.learning_starts = 10_000
        self.epsilon_timesteps = 200_000
        self.train_freq = 4

        self._loss_function = nn.SmoothL1Loss()  # huber loss # nn.MSELoss()
        self._skip_loss_function = nn.SmoothL1Loss()  # nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
        self._skip_q_optimizer = optim.Adam(self._skip_q.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)

        self._skip_replay_buffer = NoneConcatSkipReplayBuffer(1e5)  
        self._env = env
        self._eval_env = eval_env

        self.bernoulli_probability_q = 0.9
        self.bernoulli_probability_sq = 0.5
        self.uncertainty_factor = uncertainty_factor

        self._time_limt = 999999999

        # Load model
        if args.load_dir:
            load_dir = os.path.join(args.out_dir, args.load_dir)   
            self.load_model(load_dir)
        else:
            self.total_steps = 0

    def get_action(self, x: np.ndarray, header_number:int=None) -> int:
        with torch.no_grad():
            if header_number is not None:
                action = self._q(tt(x[None, :]),header_number).cpu()
                return int(action.max(1).indices.numpy())
            else:
                # vote
                actions = self._q(tt(x[None, :]))
                actions = [int(action.cpu().max(1).indices.numpy()) for action in actions]
                actions = Counter(actions)
                action = actions.most_common(1)[0][0]
                return action                

    def get_skip(self, x , a):
        current_outputs = self._skip_q(tt(x[None, :]), tt(a[None, :]))
        outputs = []
        for k in range(self.n_heads):
            outputs.append(current_outputs[k].detach().cpu().numpy())
        outputs = np.array(outputs)
        mean_Q = np.mean(outputs , axis=0) 
        std_Q = np.std(outputs, axis=0)
        Q_tilda = mean_Q + self.uncertainty_factor*std_Q
        u = np.argmax(Q_tilda)
        return u

    def get_batch_skip(self, x , a):
        current_outputs = self._skip_q(x, a)
        outputs = []
        for k in range(self.n_heads):
            outputs.append(current_outputs[k].detach().cpu().numpy())  
        outputs = np.array(outputs) 
        mean_Q = np.mean(outputs, axis=0) 
        std_Q = np.std(outputs, axis=0)
        Q_tilda = mean_Q + self.uncertainty_factor*std_Q
        u = np.argmax(Q_tilda, axis=-1) 
        return u

    def train(self, episodes: int, max_env_time_steps: int, epsilon: float, eval_eps: int = 1,
              eval_every_n_steps: int = 1, max_train_time_steps: int = 1_000_000):
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
        num_update_steps = 0
        batch_size = self.batch_size
        grad_clip_val = self.grad_clip_val
        target_net_upd_freq = self.target_net_upd_freq
        learning_starts = self.learning_starts

        start_time = time.time()

        for e in range(episodes):
            print("# Episode: %s/%s" % (e + 1, episodes))
            s = self._env.reset()
            es = 0
            # Sample Active heads
            heads = list(range(self.n_heads))
            np.random.shuffle(heads)
            active_head = heads[0]            

            for _ in count():
                a = self.get_action(s, active_head)
                skip = self.get_skip(s, np.array([a])) 

                d = False
                skip_states, skip_rewards = [], []
                for curr_skip in range(skip + 1):  # repeat the selected action for "skip" times
                    ns, r, d, info_ = self._env.step(a)
                    self.total_steps += 1
                    es += 1
                    skip_states.append(s)  # keep track of all observed skips
                    skip_rewards.append(r)

                    #### Begin Evaluation
                    if (self.total_steps % eval_every_n_steps) == 0:
                        eval_s, eval_r, eval_d = self.eval(eval_eps, max_env_time_steps)
                        eval_stats = dict(
                            elapsed_time=time.time() - start_time,
                            training_steps=self.total_steps,
                            training_eps=e,
                            avg_num_steps_per_eval_ep=float(np.round(np.mean(eval_s), 1)),
                            avg_num_decs_per_eval_ep=float(np.round(np.mean(eval_d), 1)),
                            avg_rew_per_eval_ep=float(np.round(np.mean(eval_r), 1)),
                            std_rew_per_eval_ep=float(np.round(np.std(eval_r), 1)),
                            eval_eps=eval_eps
                        )
                        print('Done %4d/%4d episodes, rewards: %4d' % (e, episodes, float(np.mean(eval_r))))
                        with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                            json.dump(eval_stats, out_fh)
                            out_fh.write('\n')
                        self.save_model(out_dir)
                    #### End Evaluation

                    # Update the skip replay buffer with all observed skips.
                    skip_id = 0
                    for start_state in skip_states:
                        skip_reward = 0
                        for exp, r in enumerate(skip_rewards[skip_id:]):  # make sure to properly discount
                            skip_reward += np.power(self._gamma, exp) * r

                        self._skip_replay_buffer.add_transition(start_state, curr_skip - skip_id, ns,
                                                                skip_reward, d, curr_skip - skip_id + 1,
                                                                np.array([a]))  # also keep track of the behavior action
                        skip_id += 1

                    # Skip Q update based on double DQN where target is behavior Q
                    if (self.total_steps > learning_starts) and (self.total_steps % self.train_freq == 0):
                        num_update_steps += 1

                        batch_states, batch_actions, batch_next_states, batch_rewards,\
                            batch_terminal_flags, batch_lengths, batch_behaviours = \
                            self._skip_replay_buffer.random_next_batch(batch_size*2)

                        # next target z ? -> bootstrapped value? or k'th value?
                        # target = batch_rewards + (1 - batch_terminal_flags) * torch.pow(self._gamma, batch_lengths) * \
                        #         self._q_target(batch_next_states)[active_head][torch.arange(batch_size*2).long(), torch.argmax(
                        #             self._q(batch_next_states)[active_head], dim=1)]
                        
                        target = batch_rewards + (1 - batch_terminal_flags) * torch.pow(self._gamma, batch_lengths) * \
                                self._q_target.forward_mean(batch_next_states)[torch.arange(batch_size*2).long(), torch.argmax(
                                    self._q.forward_mean(batch_next_states), dim=1)]

                        current_outputs = self._skip_q(batch_states, batch_behaviours)
                        masks = torch.bernoulli(torch.zeros((batch_size*2, self.n_heads), device=device) + self.bernoulli_probability_sq)
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
                        for param in self._skip_q.core_net.parameters():
                            if param.grad is None:
                                pass
                            else:
                                param.grad.data *= 1.0/float(self.n_heads)
                        for param in self._skip_q.parameters():
                            if param.grad is None:
                                pass
                            else:
                                param.grad.data.clamp_(-grad_clip_val, grad_clip_val)
                        self._skip_q_optimizer.step()


                    # nstep target
                    if (self.total_steps > learning_starts) and (self.total_steps % self.train_freq == 0):

                        batch_states, batch_skips, batch_next_states, batch_rewards,\
                                batch_terminal_flags, batch_lengths, batch_behaviours = \
                                self._skip_replay_buffer.random_next_batch(batch_size*3)

                        current_skip_prediction = self.get_batch_skip(batch_states, batch_behaviours)
                        batch_use = np.where(batch_skips.squeeze().detach().cpu().numpy() <= current_skip_prediction)[0]
                        tmp_batch_size = len(batch_use)
                        if tmp_batch_size > batch_size:
                            batch_use = batch_use[:batch_size]
                            tmp_batch_size = len(batch_use)
                        batch_states = batch_states[batch_use]
                        batch_next_states = batch_next_states[batch_use]
                        batch_rewards = batch_rewards[batch_use]
                        batch_terminal_flags = batch_terminal_flags[batch_use]
                        batch_lengths = batch_lengths[batch_use]
                        batch_behaviours = batch_behaviours[batch_use]
                        
                        ########### Begin training each head of Bootstrapped DQN
                        next_q_target = self._q_target(batch_next_states)
                        next_q = self._q(batch_next_states)
                        current_q = self._q(batch_states)
                        masks = torch.bernoulli(torch.zeros((tmp_batch_size, self.n_heads), device=device) + self.bernoulli_probability_q )
                        cnt_losses = []
                        for k in range(self.n_heads):
                            total_used = torch.sum(masks[:,k])
                            if total_used > 0.0:
                                target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                                        next_q_target[k][torch.arange(tmp_batch_size).long(), torch.argmax(
                                            next_q[k], dim=1)]
                                current_prediction = current_q[k][torch.arange(tmp_batch_size).long(), batch_behaviours.squeeze().long()]
                                l1loss = self._loss_function(current_prediction, target.detach())
                                full_loss = masks[:,k]*l1loss
                                loss = torch.sum(full_loss/total_used)
                                cnt_losses.append(loss)

                        num_update_steps += 1
                        self._q_optimizer.zero_grad()
                        loss = sum(cnt_losses)/self.n_heads
                        loss.backward()
                        for param in self._q.core_net.parameters():
                            if param.grad is None:
                                pass
                            else:
                                param.grad.data *= 1.0/float(self.n_heads)

                        for param in self._q.parameters():
                            param.grad.data.clamp_(-grad_clip_val, grad_clip_val)
                        self._q_optimizer.step()


                        if (self.total_steps % target_net_upd_freq) == 0:
                            hard_update(self._q_target, self._q)
                        # soft_update(self._q_target, self._q, 0.01)
                        ########### End double Q-learning update
                
                    if es >= max_env_time_steps or d or self.total_steps >= max_train_time_steps or ((time.time() - start_time )>self._time_limt):
                        break
                    s = ns
                if es >= max_env_time_steps or d or self.total_steps >= max_train_time_steps or ((time.time() - start_time )>self._time_limt):
                    break
            if self.total_steps >= max_train_time_steps or ((time.time() - start_time )>self._time_limt):
                break
            

    def eval(self, episodes: int, max_env_time_steps: int):
        steps, rewards, decisions = [], [], []
        with torch.no_grad():
            for e in range(episodes):
                ed, es, er = 0, 0, 0

                s = self._eval_env.reset()
                for _ in count():
                    a = self.get_action(s)
                    skip = self.get_skip(s, np.array([a]))
                    ed += 1
                    d = False

                    for _ in range(skip + 1):
                        ns, r, d, _ = self._eval_env.step(a)
                        er += r
                        es += 1
                        if es >= max_env_time_steps or d:
                            break
                        s = ns
                    if es >= max_env_time_steps or d:
                        break
                steps.append(es)
                rewards.append(er)
                decisions.append(ed)

        return steps, rewards, decisions

    def save_model(self, path):
        torch.save({
            'total_steps': self.total_steps,
            'model_state_dict': self._q.state_dict(),
            'target_state_dict': self._q_target.state_dict(),
            'skip_q_state_dict': self._skip_q.state_dict(),
            'optimizer_state_dict': self._q_optimizer.state_dict(),
            'skip_optimizer_state_dict' : self._skip_q_optimizer.state_dict()
            }, os.path.join(path, 'Q.pt'))
        self._skip_replay_buffer.save_buffer(path)

    def load_model(self, path):
        checkpoint = torch.load(os.path.join(path, 'Q.pt'))   
        self._q.load_state_dict(checkpoint['model_state_dict'])   
        self._q_target.load_state_dict(checkpoint['target_state_dict'])
        self._q_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._skip_q.load_state_dict(checkpoint['skip_q_state_dict'])
        self._skip_q_optimizer.load_state_dict(checkpoint['skip_optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self._skip_replay_buffer.load_buffer(path)

class EZ_greedy:
    """
    ez-greedy DQN based on Dabney et al. 2020 Temporally-Extended epsilon-Greedy Exploration
    """

    def __init__(self, state_dim: int, action_dim: int, skip_dim: int, u: float, gamma: float,
                 env: gym.Env, eval_env: gym.Env):
        """
        Initialize the DQN Agent
        :param state_dim: dimensionality of the input states
        :param action_dim: dimensionality of the output actions
        :param skip_dim: dimenionality of the skip output
        :param gamma: discount factor
        :param u: for zeta distribution
        :param env: environment to train on
        :param eval_env: environment to evaluate on
        :param vision: boolean flag to indicate if the input state is an image or not
        """

        self._q = NatureDQN(state_dim, action_dim).to(device)
        self._q_target = NatureDQN(state_dim, action_dim).to(device)

        self._gamma = gamma
        self._u = u

        self.batch_size = 32
        self.grad_clip_val = 40.0
        self.target_net_upd_freq = 500
        self.learning_starts = 10_000
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.01
        self.epsilon_timesteps = 200_000
        self.train_freq = 4

        self._loss_function = nn.SmoothL1Loss()  # huber loss # nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
        self._action_dim = action_dim
        self._skip_dim = skip_dim

        self._replay_buffer = ReplayBuffer(5e4)
        self._env = env
        self._eval_env = eval_env

        # Load model
        if args.load_dir:
            load_dir = os.path.join(args.out_dir, args.load_dir)   
            self.load_model(load_dir)
        else:
            self.total_steps = 0

    def get_ez_action(self, x: np.ndarray, epsilon: float, n: int, w: int):
        if n == 0:
            if random.random() < epsilon:
                n = min(int(np.random.zipf(self._u, 1)), self._skip_dim) # zeta dist
                w = np.random.randint(self._action_dim)
                action = w
            else:
                action = np.argmax(self._q(tt(x[None, :])).cpu().detach().numpy())
        else:
            action = w
            n -= 1
        return n, action

    def train(self, episodes: int, max_env_time_steps: int, epsilon: float, eval_eps: int = 1,
              eval_every_n_steps: int = 1, max_train_time_steps: int = 1_000_000):
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
        dir_comp = out_dir.split('/')
        dir_comp[3] = 'naive'
        naive_dir = '/'.join(dir_comp)

        os.makedirs(os.path.join(os.getcwd(), naive_dir))
        num_update_steps = 0
        batch_size = self.batch_size
        grad_clip_val = self.grad_clip_val
        target_net_upd_freq = self.target_net_upd_freq
        learning_starts = self.learning_starts

        start_time = time.time()

        for e in range(episodes):
            print("# Episode: %s/%s" % (e + 1, episodes))
            s = self._env.reset()
            # for ez-greedy
            n = 0
            a = 0

            for t in range(max_env_time_steps):
                # s = s
                # s = torch.from_numpy(s).unsqueeze(0)
                if self.total_steps > self.epsilon_timesteps:
                    epsilon = self.final_epsilon
                else:
                    epsilon = self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * (
                                self.total_steps / self.epsilon_timesteps)

                n, a = self.get_ez_action(s, epsilon, n, a)

                ns, r, d, _ = self._env.step(a)
                self.total_steps += 1

                ########### Begin Evaluation
                if (self.total_steps % eval_every_n_steps) == 0:
                    eval_s, eval_r, eval_d = self.eval(eval_eps, max_env_time_steps, epsilon=0.001)
                    eval_stats = dict(
                        elapsed_time=time.time() - start_time,
                        training_steps=self.total_steps,
                        training_eps=e,
                        avg_num_steps_per_eval_ep=float(np.round(np.mean(eval_s), 1)),
                        avg_num_decs_per_eval_ep=float(np.round(np.mean(eval_d), 1)),
                        avg_rew_per_eval_ep=float(np.round(np.mean(eval_r), 1)),
                        std_rew_per_eval_ep=float(np.round(np.std(eval_r), 1)),
                        eval_eps=eval_eps
                    )
                    print('Done %4d/%4d episodes, rewards: %4d' % (e, episodes, float(np.mean(eval_r))))
                    with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                        json.dump(eval_stats, out_fh)
                        out_fh.write('\n')     
                    self.save_model(out_dir)
                ########### End Evaluation  ###########

                # Update replay buffer
                self._replay_buffer.add_transition(s, a, ns, r, d)

                batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                    self._replay_buffer.random_next_batch(batch_size)

                ########### Begin double Q-learning update
                target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                         self._q_target(batch_next_states)[torch.arange(batch_size).long(), torch.argmax(
                             self._q(batch_next_states), dim=1)]
                current_prediction = self._q(batch_states)[torch.arange(batch_size).long(), batch_actions.long()]

                loss = self._loss_function(current_prediction, target.detach())

                if (self.total_steps > learning_starts) and (self.total_steps % self.train_freq == 0):
                    num_update_steps += 1
                    self._q_optimizer.zero_grad()
                    loss.backward()
                    for param in self._q.parameters():
                        param.grad.data.clamp_(-grad_clip_val, grad_clip_val)
                    self._q_optimizer.step()


                    if (self.total_steps % target_net_upd_freq) == 0:
                        hard_update(self._q_target, self._q)
                    # soft_update(self._q_target, self._q, 0.01)
                    ########### End double Q-learning update
                
                if d:
                    break
                s = ns
                if self.total_steps >= max_train_time_steps:
                    break
            if self.total_steps >= max_train_time_steps:
                break
            
    def eval(self, episodes: int, max_env_time_steps: int, epsilon:float):
        steps, rewards, decisions = [], [], []
        with torch.no_grad():
            for e in range(episodes):
                ed, es, er = 0, 0, 0
                n, a = 0, 0

                s = self._eval_env.reset()
                for _ in count():
                    if n == 0:
                        ed += 1
                    n, a = self.get_ez_action(s, epsilon, n, a)
                    ns, r, d, _ = self._eval_env.step(a)
                    er += r
                    es += 1
                    if es >= max_env_time_steps or d:
                        break
                    s = ns
                steps.append(es)
                rewards.append(er)
                decisions.append(ed)

        return steps, rewards, decisions

    def save_model(self, path):
        torch.save(self._q.state_dict(), os.path.join(path, 'Q.pt'))
        torch.save({
            'total_steps': self.total_steps,
            'model_state_dict': self._q.state_dict(),
            'target_state_dict': self._q_target.state_dict(),
            'optimizer_state_dict': self._q_optimizer.state_dict(),
            }, os.path.join(path, 'Q.pt'))
        self._replay_buffer.save_buffer(path)

    def load_model(self, path):
        checkpoint = torch.load(os.path.join(path, 'Q.pt'))   
        self._q.load_state_dict(checkpoint['model_state_dict'])   
        self._q_target.load_state_dict(checkpoint['target_state_dict'])
        self._q_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self._replay_buffer.load_buffer(path)

class Fixed_J:
    """
    Double DQN Agent with a fixed skip
    """
    def __init__(self, state_dim: int, action_dim: int, skip_dim: int, u: float, gamma: float,
                 env: gym.Env, eval_env: gym.Env):

        self._q = NatureDQN(state_dim, action_dim).to(device)
        self._q_target = NatureDQN(state_dim, action_dim).to(device)

        self._gamma = gamma
        self._u = u

        self.batch_size = 32
        self.grad_clip_val = 40.0
        self.target_net_upd_freq = 500
        self.learning_starts = 10_000
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.01
        self.epsilon_timesteps = 200_000
        self.train_freq = 4

        self._loss_function = nn.SmoothL1Loss()  # huber loss # nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
        self._action_dim = action_dim
        self._skip_dim = skip_dim

        self._replay_buffer = ReplayBuffer(5e4)
        self._env = env
        self._eval_env = eval_env

        self.ext_length = 4 # fixed skip length

        # Load model
        if args.load_dir:
            load_dir = os.path.join(args.out_dir, args.load_dir)   
            self.load_model(load_dir)
        else:
            self.total_steps = 0
    
    def get_action(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(tt(x[None, :])).cpu().detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def train(self, episodes: int, max_env_time_steps: int, epsilon: float, eval_eps: int = 1,
              eval_every_n_steps: int = 1, max_train_time_steps: int = 1_000_000):
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
        num_update_steps = 0
        batch_size = self.batch_size
        grad_clip_val = self.grad_clip_val
        target_net_upd_freq = self.target_net_upd_freq
        learning_starts = self.learning_starts

        start_time = time.time()

        for e in range(episodes):
            print("# Episode: %s/%s" % (e + 1, episodes))
            s = self._env.reset()

            for t in range(max_env_time_steps):
                if self.total_steps > self.epsilon_timesteps:
                    epsilon = self.final_epsilon
                else:
                    epsilon = self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * (
                                self.total_steps / self.epsilon_timesteps)

                n, a = self.ext_length, self.get_action(s, epsilon)

                n -= 1
                if n == 0:
                    a = self.get_action(s, epsilon)
                    n = self.ext_length
                else:
                    pass # continue same action

                ns, r, d, _ = self._env.step(a)
                self.total_steps += 1

                ########### End Evaluation TEE and start naive ###########
                if (self.total_steps % eval_every_n_steps) == 0:
                    eval_s, eval_r, eval_d = self.naive_eval(eval_eps, max_env_time_steps, epsilon=0.001)
                    eval_stats = dict(
                        elapsed_time=time.time() - start_time,
                        training_steps=self.total_steps,
                        training_eps=e,
                        avg_num_steps_per_eval_ep=float(np.round(np.mean(eval_s), 1)),
                        avg_num_decs_per_eval_ep=float(np.round(np.mean(eval_d), 1)),
                        avg_rew_per_eval_ep=float(np.round(np.mean(eval_r), 1)),
                        std_rew_per_eval_ep=float(np.round(np.std(eval_r), 1)),
                        eval_eps=eval_eps
                    )
                    print('Done %4d/%4d episodes, rewards: %4d' % (e, episodes, float(np.mean(eval_r))))
                    with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                        json.dump(eval_stats, out_fh)
                        out_fh.write('\n')     
                    self.save_model(out_dir)
                    ########### End Evaluation

                # Update replay buffer
                self._replay_buffer.add_transition(s, a, ns, r, d)

                batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                    self._replay_buffer.random_next_batch(batch_size)

                ########### Begin double Q-learning update
                target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                         self._q_target(batch_next_states)[torch.arange(batch_size).long(), torch.argmax(
                             self._q(batch_next_states), dim=1)]
                current_prediction = self._q(batch_states)[torch.arange(batch_size).long(), batch_actions.long()]

                loss = self._loss_function(current_prediction, target.detach())

                if (self.total_steps > learning_starts) and (self.total_steps % self.train_freq == 0):
                    num_update_steps += 1
                    self._q_optimizer.zero_grad()
                    loss.backward()
                    for param in self._q.parameters():
                        param.grad.data.clamp_(-grad_clip_val, grad_clip_val)
                    self._q_optimizer.step()


                    if (self.total_steps % target_net_upd_freq) == 0:
                        hard_update(self._q_target, self._q)
                    # soft_update(self._q_target, self._q, 0.01)
                    ########### End double Q-learning update
                
                if d:
                    break
                s = ns
                if self.total_steps >= max_train_time_steps:
                    break
            if self.total_steps >= max_train_time_steps:
                break

    def naive_eval(self, episodes: int, max_env_time_steps: int, epsilon:float):
        steps, rewards, decisions = [], [], []
        with torch.no_grad():
            for e in range(episodes):
                ed, es, er = 1, 0, 0
                s = self._eval_env.reset()
                n, a = self.ext_length, self.get_action(s, epsilon)

                for _ in count():
                    n -= 1
                    if n == 0:
                        ed += 1
                        a = self.get_action(s, epsilon)
                        n = self.ext_length
                    else:
                        pass # continue same action

                    ns, r, d, _ = self._eval_env.step(a)
                    er += r
                    es += 1
                    if es >= max_env_time_steps or d:
                        break
                    s = ns
                steps.append(es)
                rewards.append(er)
                decisions.append(ed)

        return steps, rewards, decisions

    def save_model(self, path):
        torch.save(self._q.state_dict(), os.path.join(path, 'Q.pt'))
        torch.save({
            'total_steps': self.total_steps,
            'model_state_dict': self._q.state_dict(),
            'target_state_dict': self._q_target.state_dict(),
            'optimizer_state_dict': self._q_optimizer.state_dict(),
            }, os.path.join(path, 'Q.pt'))
        self._replay_buffer.save_buffer(path)

    def load_model(self, path):
        checkpoint = torch.load(os.path.join(path, 'Q.pt'))   
        self._q.load_state_dict(checkpoint['model_state_dict'])   
        self._q_target.load_state_dict(checkpoint['target_state_dict'])
        self._q_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self._replay_buffer.load_buffer(path)

class Naive_Option_DQN:
    """
    Ablation study to evaluate the effect of option decomposition, the size of the action space : |A|x|J|
    """

    def __init__(self, state_dim: int, action_dim: int,
                 num_output_duplication: int, skip_map: dict,
                 gamma: float, env: gym.Env, eval_env: gym.Env):
        
        self._q = NatureDQN(state_dim, action_dim * num_output_duplication).to(device)
        self._q_target = NatureDQN(state_dim, action_dim * num_output_duplication).to(device)

        self._gamma = gamma

        self.batch_size = 32
        self.grad_clip_val = 40.0
        self.target_net_upd_freq = 500
        self.learning_starts = 10_000
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.01
        self.epsilon_timesteps = 200_000
        self.train_freq = 4   

        self._loss_function = nn.SmoothL1Loss()  # huber loss # nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
        self._action_dim = action_dim

        self._replay_buffer = ReplayBuffer(5e4)
        self._skip_map = skip_map
        self._dup_vals = num_output_duplication   
        self._env = env
        self._eval_env = eval_env

        # Load model
        if args.load_dir:
            load_dir = os.path.join(args.out_dir, args.load_dir)   
            self.load_model(load_dir)
        else:
            self.total_steps = 0

    def get_action(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(tt(x[None, :])).detach().cpu().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim*self._dup_vals)
        return u

    def train(self, episodes: int, max_env_time_steps: int, epsilon: float, eval_eps: int = 1,
              eval_every_n_steps: int = 1, max_train_time_steps: int = 1_000_000):
        """
        Training loop
        :param episodes: maximum number of episodes to train for
        :param max_env_time_steps: maximum number of steps in the environment to perform per episode
        :param epsilon: constant epsilon for exploration when selecting actions
        :param eval_eps: numper of episodes to run for evaluation
        :param eval_every_n_steps: interval of steps after which to evaluate the trained agent
        :param max_train_time_steps: maximum number of steps to train
        """
        num_update_steps = 0
        batch_size = self.batch_size
        grad_clip_val = self.grad_clip_val
        target_net_upd_freq = self.target_net_upd_freq
        learning_starts = self.learning_starts

        start_time = time.time()
        for e in range(episodes):
            print("%s/%s" % (e + 1, episodes))
            s = self._env.reset()
            es = 0
            for t in range(max_env_time_steps):
                if self.total_steps > self.epsilon_timesteps:
                    epsilon = self.final_epsilon
                else:
                    epsilon = self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * (
                            self.total_steps / self.epsilon_timesteps)
                a = self.get_action(s, epsilon)

                # convert action id int corresponding behaviour action and skip value
                act = a // self._dup_vals  # behaviour
                rep = a // self._env.action_space.n  # skip id
                skip = self._skip_map[rep]  # skip id to corresponding skip value

                d = False
                skip_states, skip_rewards = [], []
                for _ in range(skip + 1):  # repeat chosen behaviour action for "skip" steps
                    ns, r, d, _ = self._env.step(act)
                    self.total_steps += 1
                    es += 1
                    skip_states.append(s)  # keep track of all observed skips
                    skip_rewards.append(r)
                    
                    # Update the skip replay buffer with all observed skips.
                    skip_id = 0
                    for start_state in skip_states:
                        skip_reward = 0
                        for exp, r in enumerate(skip_rewards[skip_id:]):  # make sure to properly discount
                            skip_reward += np.power(self._gamma, exp) * r
                        self._replay_buffer.add_transition(s, a - skip_id, ns, r, d)  # also keep track of the behavior action
                        skip_id += 1

                    ########### Begin Evaluation
                    if (self.total_steps % eval_every_n_steps) == 0:
                        eval_s, eval_r, eval_d = self.eval(eval_eps, max_env_time_steps, epsilon=0.001)
                        eval_stats = dict(
                            elapsed_time=time.time() - start_time,
                            training_steps=self.total_steps,
                            training_eps=e,
                            avg_num_steps_per_eval_ep=float(np.round(np.mean(eval_s), 1)),
                            avg_num_decs_per_eval_ep=float(np.round(np.mean(eval_d), 1)),
                            avg_rew_per_eval_ep=float(np.round(np.mean(eval_r), 1)),
                            std_rew_per_eval_ep=float(np.round(np.std(eval_r), 1)),
                            eval_eps=eval_eps
                        )
                        print('Done %4d/%4d episodes, rewards: %4d' % (e, episodes, float(np.mean(eval_r))))
                        with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                            json.dump(eval_stats, out_fh)
                            out_fh.write('\n')
                            self.save_model(out_dir)
                    ########### End Evaluation
                    
                    ### Q-update based double Q learning

                    batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                        self._replay_buffer.random_next_batch(batch_size)

                    target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                             self._q_target(batch_next_states)[torch.arange(batch_size).long(), torch.argmax(
                                 self._q(batch_next_states), dim=1)]
                    current_prediction = self._q(batch_states)[torch.arange(batch_size).long(), batch_actions.long()]

                    loss = self._loss_function(current_prediction, target.detach())
                    if (self.total_steps > learning_starts) and (self.total_steps % self.train_freq == 0):
                        num_update_steps += 1
                        self._q_optimizer.zero_grad()
                        loss.backward()
                        for param in self._q.parameters():
                            param.grad.data.clamp_(-grad_clip_val, grad_clip_val)
                        self._q_optimizer.step()

                        if (self.total_steps % target_net_upd_freq) == 0:
                            hard_update(self._q_target, self._q)
                        # soft_update(self._q_target, self._q, 0.01)
                        ########### End double Q-learning update
                    
                    if es >= max_env_time_steps or d or self.total_steps >= max_train_time_steps:
                        break

                    s = ns
                if es >= max_env_time_steps or d or self.total_steps >= max_train_time_steps:
                    break

            if self.total_steps >= max_train_time_steps:
                break

    def eval(self, episodes: int, max_env_time_steps: int, epsilon:float):
        """
        Simple method that evaluates the agent with fixed epsilon = 0
        :param episodes: max number of episodes to play
        :param max_env_time_steps: max number of max_env_time_steps to play
        :returns (steps per episode), (reward per episode), (decisions per episode)
        """
        steps, rewards, decisions = [], [], []
        with torch.no_grad():
            for e in range(episodes):
                ed, es, er = 0, 0, 0

                s = self._eval_env.reset()
                for _ in count():
                    a = self.get_action(s, epsilon)
                    act = a // self._dup_vals # behaviour
                    rep = a // self._env.action_space.n  # skip id
                    skip = self._skip_map[rep]

                    ed += 1

                    d = False
                    for _ in range(skip + 1):
                        ns, r, d, _ = self._eval_env.step(act)
                        er += r
                        es += 1
                        if es >= max_env_time_steps or d:
                            break
                        s = ns
                    if es >= max_env_time_steps or d:
                        break
                steps.append(es)
                rewards.append(er)
                decisions.append(ed)

        return steps, rewards, decisions

    def save_model(self, path):
        torch.save(self._q.state_dict(), os.path.join(path, 'Q.pt'))
        torch.save({
            'total_steps': self.total_steps,
            'model_state_dict': self._q.state_dict(),
            'target_state_dict': self._q_target.state_dict(),
            'optimizer_state_dict': self._q_optimizer.state_dict(),
            }, os.path.join(path, 'Q.pt'))
        self._replay_buffer.save_buffer(path)

    def load_model(self, path):
        checkpoint = torch.load(os.path.join(path, 'Q.pt'))   
        self._q.load_state_dict(checkpoint['model_state_dict'])   
        self._q_target.load_state_dict(checkpoint['target_state_dict'])
        self._q_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self._replay_buffer.load_buffer(path)


class UTE_Bandit:
    """
    UTE with Adpative Uncertainty Parameter (lambda) Controlled by Bandit
    """

    def __init__(self, state_dim, action_dim, skip_dim, gamma, env, eval_env, window_size, ucb_epsilon, ucb_beta):

        self.core_net = Extension_CoreNet()
        self.core_net_target = Extension_CoreNet()
        self._q = NatureDQNhead(self.core_net, state_dim, action_dim).to(device)
        self._q_target = NatureDQNhead(self.core_net_target, state_dim, action_dim).to(device)       

        self.n_heads = 10
        self._skip_q = Ensemble_Extension(self.core_net, state_dim, skip_dim, self.n_heads).to(device)
        print('Using {} as Q'.format(str(self._q)))
        print('Using {} as skip-Q\n{}'.format(str(self._skip_q), '#' * 80))

        self._gamma = gamma
        self._action_dim = action_dim
        self._skip_dim = skip_dim

        self.batch_size = 32
        self.grad_clip_val = 40.0
        self.target_net_upd_freq = 500
        self.learning_starts = 10_000
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.01
        self.epsilon_timesteps = 200_000
        self.train_freq = 4
        self.bernoulli_probability = 0.5

        # For bandit: Adaptively choose uncertainty factor lambda
        self.lambdas = [-1.5, -1.0, -0.5, -0.2, 0.0, +0.2, +0.5, +1.0, +1.5]
        num_arms = len(self.lambdas)
        self.ucb = UCB(num_arms, window_size, ucb_epsilon, ucb_beta)

        self._loss_function = nn.SmoothL1Loss()  # huber loss # nn.MSELoss()
        self._skip_loss_function = nn.SmoothL1Loss()  # nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
        self._skip_q_optimizer = optim.Adam(self._skip_q.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
        
        self._skip_replay_buffer = NoneConcatSkipReplayBuffer(1e5)
        self._env = env
        self._eval_env = eval_env

        self._time_limt = 42500

        # Load model
        if args.load_dir:
            load_dir = os.path.join(args.out_dir, args.load_dir)   
            self.load_model(load_dir)
        else:
            self.total_steps = 0

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
        current_outputs = self._skip_q(tt(x[None, :]), tt(a[None, :]))
        outputs = []
        for k in range(self.n_heads):
            outputs.append(current_outputs[k].detach().cpu().numpy())
        outputs = np.array(outputs)
        mean_Q = np.mean(outputs , axis=0) # 1x10
        std_Q = np.std(outputs, axis=0)
        Q_tilda = mean_Q + self.uncertainty_factor*std_Q
        u = np.argmax(Q_tilda)
        return u

    def get_batch_skip(self, x , a):
        current_outputs = self._skip_q(x, a)
        outputs = []
        for k in range(self.n_heads):
            outputs.append(current_outputs[k].detach().cpu().numpy())  # Bx10
        outputs = np.array(outputs) # HxBx10
        mean_Q = np.mean(outputs, axis=0) # Bx10
        std_Q = np.std(outputs, axis=0)
        Q_tilda = mean_Q + self.uncertainty_factor*std_Q
        u = np.argmax(Q_tilda, axis=-1) #B
        return u

    def train(self, episodes: int, max_env_time_steps: int, epsilon: float, eval_eps: int = 1,
              eval_every_n_steps: int = 1, max_train_time_steps: int = 1_000_000):
        """
        Training loop
        :param episodes: maximum number of episodes to train for
        :param max_env_time_steps: maximum number of steps in the environment to perform per episode
        :param epsilon: constant epsilon for exploration when selecting actions
        :param eval_eps: numper of episodes to run for evaluation
        :param eval_every_n_steps: interval of steps after which to evaluate the trained agent
        :param max_train_time_steps: maximum number of steps to train
        """
        num_update_steps = 0
        batch_size = self.batch_size
        grad_clip_val = self.grad_clip_val
        target_net_upd_freq = self.target_net_upd_freq
        learning_starts = self.learning_starts

        start_time = time.time()

        for e in range(episodes):
            print("# Episode: %s/%s" % (e + 1, episodes))

            # get index from ucb
            j = self.ucb.pull_index()

            # get beta gamma
            self.uncertainty_factor = self.lambdas[j]

            ucb_datas = []

            s = self._env.reset()
            es = 0

            for _ in count():

                if self.total_steps > self.epsilon_timesteps:
                    epsilon = self.final_epsilon
                else:
                    epsilon = self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * (
                                self.total_steps / self.epsilon_timesteps)

                a = self.get_action(s, epsilon)
                skip = self.get_skip(s, np.array([a]))  # get skip with the selected action as context

                d = False
                skip_states, skip_rewards = [], []
                for curr_skip in range(skip + 1):  # repeat the selected action for "skip" times
                    ns, r, d, info_ = self._env.step(a)
                    self.total_steps += 1
                    es += 1
                    skip_states.append(s)  # keep track of all observed skips
                    skip_rewards.append(r)
                    ucb_datas.append((j, r))

                    #### Begin Evaluation
                    if (self.total_steps % eval_every_n_steps) == 0:
                        eval_s, eval_r, eval_d = self.eval(eval_eps, max_env_time_steps, epsilon=0.001)
                        eval_stats = dict(
                            elapsed_time=time.time() - start_time,
                            training_steps=self.total_steps,
                            training_eps=e,
                            avg_num_steps_per_eval_ep=float(np.round(np.mean(eval_s), 1)),
                            avg_num_decs_per_eval_ep=float(np.round(np.mean(eval_d), 1)),
                            avg_rew_per_eval_ep=float(np.round(np.mean(eval_r), 1)),
                            std_rew_per_eval_ep=float(np.round(np.std(eval_r), 1)),
                            eval_eps=eval_eps, 
                            uncertainty_factor=self.uncertainty_factor
                        )
                        print('Done %4d/%4d episodes, rewards: %4d' % (e, episodes, float(np.mean(eval_r))))
                        with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                            json.dump(eval_stats, out_fh)
                            out_fh.write('\n')
                        self.save_model(out_dir)
                    #### End Evaluation

                    # Update the skip replay buffer with all observed skips.
                    skip_id = 0
                    for start_state in skip_states:
                        skip_reward = 0
                        for exp, r in enumerate(skip_rewards[skip_id:]):  # make sure to properly discount
                            skip_reward += np.power(self._gamma, exp) * r

                        self._skip_replay_buffer.add_transition(start_state, curr_skip - skip_id, ns,
                                                                skip_reward, d, curr_skip - skip_id + 1,
                                                                np.array([a]))  # also keep track of the behavior action
                        skip_id += 1

                    # Skip Q update based on double DQN where target is behavior Q
                    if (self.total_steps > learning_starts) and (self.total_steps % self.train_freq == 0):
                        num_update_steps += 1

                        batch_states, batch_actions, batch_next_states, batch_rewards,\
                            batch_terminal_flags, batch_lengths, batch_behaviours = \
                            self._skip_replay_buffer.random_next_batch(batch_size*2)

                        target = batch_rewards + (1 - batch_terminal_flags) * torch.pow(self._gamma, batch_lengths) * \
                                self._q_target(batch_next_states)[torch.arange(batch_size*2).long(), torch.argmax(
                                    self._q(batch_next_states), dim=1)]
                        
                        current_outputs = self._skip_q(batch_states, batch_behaviours)
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
                        for param in self._skip_q.core_net.parameters():
                            if param.grad is None:
                                pass
                                #print("##### Skip Q Parameter with grad = None:", param.name)
                            else:
                                param.grad.data *= 1.0/float(self.n_heads)
                        for param in self._skip_q.parameters():
                            if param.grad is None:
                                pass
                                #print("##### Skip Q Parameter with grad = None:", param.name)
                            else:
                                param.grad.data.clamp_(-grad_clip_val, grad_clip_val)
                        self._skip_q_optimizer.step()

                    
                    # Action Q update based on double DQN with nstep target
                    if (self.total_steps > learning_starts) and (self.total_steps % self.train_freq == 0):

                        batch_states, batch_skips, batch_next_states, batch_rewards,\
                            batch_terminal_flags, batch_lengths, batch_behaviours = \
                            self._skip_replay_buffer.random_next_batch(batch_size*3)

                        current_skip_prediction = self.get_batch_skip(batch_states, batch_behaviours) 
                        batch_use = np.where(batch_skips.squeeze().detach().cpu().numpy() <= current_skip_prediction)[0]
                        tmp_batch_size = len(batch_use)
                        if tmp_batch_size > batch_size:
                            batch_use = batch_use[:batch_size]
                            tmp_batch_size = len(batch_use)

                        batch_states = batch_states[batch_use]
                        batch_next_states = batch_next_states[batch_use]
                        batch_rewards = batch_rewards[batch_use]
                        batch_terminal_flags = batch_terminal_flags[batch_use]
                        batch_lengths = batch_lengths[batch_use]
                        batch_behaviours = batch_behaviours[batch_use]

                        #residual_skip = (current_skip_prediction - batch_skips.squeeze())[batch_use]
                        next_actions = torch.argmax(self._q(batch_next_states), dim=1)
                        #next_actions = torch.where(residual_skip==0, next_actions, batch_behaviours.squeeze().long())

                        target = batch_rewards + (1 - batch_terminal_flags) * torch.pow(self._gamma, batch_lengths)* \
                                self._q_target(batch_next_states)[torch.arange(tmp_batch_size).long(), next_actions]
                        current_prediction = self._q(batch_states)[torch.arange(tmp_batch_size).long(), batch_behaviours.squeeze().long()]
                        loss = self._loss_function(current_prediction, target.detach())

                        self._q_optimizer.zero_grad()
                        loss.backward()
                        for param in self._q.parameters():
                            if param.grad is None:
                                pass
                                # print("##### Q Parameter with grad = None:", param.name)
                            else:
                                param.grad.data.clamp_(-grad_clip_val, grad_clip_val)
                        self._q_optimizer.step()

                        if (self.total_steps % target_net_upd_freq) == 0:
                            hard_update(self._q_target, self._q)
                            # print("DQN loss: {} SkipQ loss: {}".format(loss, skip_loss))
                        # soft_update(self._q_target, self._q, 0.01)

                    if es >= max_env_time_steps or d or self.total_steps >= max_train_time_steps or ((time.time() - start_time )>self._time_limt):
                        break
                    s = ns
                if es >= max_env_time_steps or d or self.total_steps >= max_train_time_steps or ((time.time() - start_time )>self._time_limt):
                    break

            # Update UCB
            self.ucb.push_data(ucb_datas)
            if self.total_steps >= max_train_time_steps or ((time.time() - start_time )>self._time_limt):
                break

    def eval(self, episodes: int, max_env_time_steps: int, epsilon: float):
        """
        Simple method that evaluates the agent
        :param episodes: max number of episodes to play
        :param max_env_time_steps: max number of max_env_time_steps to play
        :returns (steps per episode), (reward per episode), (decisions per episode)
        """
        steps, rewards, decisions = [], [], []
        with torch.no_grad():
            for e in range(episodes):
                ed, es, er = 0, 0, 0

                s = self._eval_env.reset()
                for _ in count():
                    a = self.get_action(s, epsilon)
                    skip = self.get_skip(s, np.array([a]))
                    ed += 1
                    d = False
                    for _ in range(skip + 1):
                        ns, r, d, _ = self._eval_env.step(a)
                        er += r
                        es += 1
                        if es >= max_env_time_steps or d:
                            break
                        s = ns
                    if es >= max_env_time_steps or d:
                        break
                steps.append(es)
                rewards.append(er)
                decisions.append(ed)

        return steps, rewards, decisions

    def save_model(self, path):
        torch.save({
            'total_steps': self.total_steps,
            'model_state_dict': self._q.state_dict(),
            'target_state_dict': self._q_target.state_dict(),
            'skip_q_state_dict': self._skip_q.state_dict(),
            'optimizer_state_dict': self._q_optimizer.state_dict(),
            'skip_optimizer_state_dict' : self._skip_q_optimizer.state_dict()
            }, os.path.join(path, 'Q.pt'))
        self._skip_replay_buffer.save_buffer(path)

    def load_model(self, path):
        checkpoint = torch.load(os.path.join(path, 'Q.pt'))   
        self._q.load_state_dict(checkpoint['model_state_dict'])   
        self._q_target.load_state_dict(checkpoint['target_state_dict'])
        self._q_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._skip_q.load_state_dict(checkpoint['skip_q_state_dict'])
        self._skip_q_optimizer.load_state_dict(checkpoint['skip_optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self._skip_replay_buffer.load_buffer(path)



if __name__ == "__main__":
    import argparse

    outdir_suffix_dict = {'none': '', 'empty': '', 'time': '%Y_%m_%d_%H%M%S',
                          'seed': '{:d}', 'params': '{:d}_{:d}_{:d}',
                          'paramsseed': '{:d}_{:d}_{:d}_{:d}'}
    parser = argparse.ArgumentParser('TempoRL')
    parser.add_argument('--episodes', '-e',
                        default=100,
                        type=int,
                        help='Number of training episodes.')
    parser.add_argument('--training-steps', '-t',
                        default=1_000_000,
                        type=int,
                        help='Number of training episodes.')
    parser.add_argument('--out-dir',
                        default=None,
                        type=str,
                        help='Directory to save results. Defaults to tmp dir.')
    parser.add_argument('--out-dir-suffix',
                        default='paramsseed',
                        type=str,
                        choices=list(outdir_suffix_dict.keys()),
                        help='Created suffix of directory to save results.')
    parser.add_argument('--seed', '-s',
                        default=12345,
                        type=int,
                        help='Seed')
    parser.add_argument('--eval-after-n-steps',
                        default=10 ** 3,
                        type=int,
                        help='After how many steps to evaluate')
    parser.add_argument('--eval-n-episodes',
                        default=1,
                        type=int,
                        help='How many episodes to evaluate')             

    # DQN -> normal double DQN agent
    # DAR -> Dynamic action repetition agent based on normal DDQN with repeated output heads for different skip values
    # tdqn -> TempoRL DDQN with shared state representation for behaviour and skip Qs
    # ute_onestep -> UTE without n-step learning
    # ute -> UTE
    # bootstrap -> Bootstrapped DQN
    # ez-greedy -> ez-greedy DQN 
    # full_ensemble -> UTE + Bootstrapped DQN desicribed in Appendix
    # fixed_j -> Fixed Repeat agent
    # o_dqn -> Undecomposed action repetition agent
    parser.add_argument('--agent',
                        choices=['dqn', 'dar', 'tdqn', 'ute_onestep', 'ute', 'bootstrap',
                            'ez-greedy', 'full_ensemble', 'fixed_j', 'o_dqn', 'ute_bandit'],
                        type=str.lower,
                        help='Which agent to train',
                        default='tdqn')
    parser.add_argument('--skip-net-max-skips',
                        type=int,
                        default=10,
                        help='Maximum skip-size')
    parser.add_argument('--env-max-steps',
                        default=10000,
                        type=int,
                        help='Maximal steps in environment before termination.',
                        dest='env_ms')
    parser.add_argument('--no-frame-skip', action='store_true')
    parser.add_argument('--84x84', action='store_true', dest='large_image')
    parser.add_argument('--dar-A', default=1, type=int)
    parser.add_argument('--dar-B', default=10, type=int)
    parser.add_argument('--uncertainty-factor', default=-0.5, type=float)

    parser.add_argument('--env',
                        type=str,
                        help="Any Atari env",
                        default='qbert')
    parser.add_argument("--load-dir", default=None, type=str)

    # for TEE
    parser.add_argument('--u', default=2.0, type=int)

    # for UCB
    parser.add_argument('--window_size', default=200, type=int)
    parser.add_argument('--ucb_epsilon', default=0.2, type=float)
    parser.add_argument('--ucb_beta', default=1.0, type=float) 

    # setup output dir
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    outdir_suffix_dict['seed'] = outdir_suffix_dict['seed'].format(args.seed)
    epis = args.episodes if args.episodes else -1
    outdir_suffix_dict['params'] = outdir_suffix_dict['params'].format(
        epis, args.skip_net_max_skips, args.env_ms)
    outdir_suffix_dict['paramsseed'] = outdir_suffix_dict['paramsseed'].format(
        epis, args.skip_net_max_skips, args.env_ms, args.seed)


    if args.load_dir:
        out_dir = os.path.join(args.out_dir, args.load_dir) 
    else:
        out_dir = experiments.prepare_output_dir(args, user_specified_dir=args.out_dir,
                                             time_format=outdir_suffix_dict[args.out_dir_suffix])

    # Setup Envs
    game = ''.join([g.capitalize() for g in args.env.split('_')])

    if args.no_frame_skip:
        eval_game = '{}NoFrameskip-v4'.format(game)
        game = '{}NoFrameskip-v0'.format(game)
        env = make_env_old(game, dim=84 if args.large_image else 42)
        eval_env = make_env_old(eval_game, dim=84 if args.large_image else 42)
    else:
        eval_game = '{}Deterministic-v4'.format(game)
        game = '{}Deterministic-v0'.format(game)
        env = make_env(game, dim=84 if args.large_image else 42)
        eval_env = make_env(eval_game, dim=84 if args.large_image else 42)

    # Setup Agent
    state_dim = env.observation_space.shape[0]  # (4, 42, 42) or (4, 84, 84) for PyTorch order
    action_dim = env.action_space.n
    if args.agent == 'dqn':
        agent = DQN(state_dim, action_dim, gamma=0.99, env=env, eval_env=eval_env)
    elif args.agent == 'tdqn':
        agent = TDQN(state_dim, action_dim, args.skip_net_max_skips, gamma=0.99, env=env, eval_env=eval_env)
    elif args.agent == 'dar':
        if args.dar_A is not None and args.dar_B is not None:
            skip_map = {0: args.dar_A, 1: args.dar_B}
            num_output_duplication = 2
        else:
            skip_map = {a: a for a in range(args.skip_net_max_skips)}
            num_output_duplication = args.skip_net_max_skips,
        agent = DAR(state_dim, action_dim, num_output_duplication, skip_map, gamma=0.99, env=env,
                    eval_env=eval_env)
    elif args.agent =='ute_onestep':
        agent = UTE_One_Step(state_dim, action_dim, args.skip_net_max_skips, gamma=0.99, env=env,
                        eval_env=eval_env, uncertainty_factor = args.uncertainty_factor)  
    elif args.agent =='ute':
        agent = UTE(state_dim, action_dim, args.skip_net_max_skips, gamma=0.99, env=env,
                        eval_env=eval_env, uncertainty_factor = args.uncertainty_factor) 
    elif args.agent =='bootstrap':
        agent = BootstrappedDQN(state_dim, action_dim, gamma=0.99, env=env, eval_env=eval_env)                    
    elif args.agent == 'ez-greedy':
        agent = EZ_greedy(state_dim, action_dim, args.skip_net_max_skips, args.u,  gamma=0.99, env=env, eval_env=eval_env) 
    elif args.agent == 'fixed_j':
        agent = Fixed_J(state_dim, action_dim, args.skip_net_max_skips, args.u,  gamma=0.99, env=env, eval_env=eval_env)                
    elif args.agent =='full_ensemble':
        agent = Full_Ensemble(state_dim, action_dim, args.skip_net_max_skips, gamma=0.99, env=env,
                        eval_env=eval_env, uncertainty_factor = args.uncertainty_factor)    
    elif args.agent == 'o_dqn':
        skip_map = {a: a for a in range(args.skip_net_max_skips)}
        agent = Naive_Option_DQN(state_dim, action_dim, args.skip_net_max_skips, skip_map, gamma=0.99, env=env,
                    eval_env=eval_env)
    elif args.agent == 'ute_bandit':
        agent = UTE_Bandit(state_dim, action_dim, args.skip_net_max_skips, gamma=0.99, env=env, 
                        eval_env=eval_env, window_size=args.window_size, ucb_epsilon=args.ucb_epsilon, ucb_beta=args.ucb_beta) 

    episodes = args.episodes
    max_env_time_steps = args.env_ms
    epsilon = 0.2

    agent.train(episodes, max_env_time_steps, epsilon, args.eval_n_episodes, args.eval_after_n_steps,
                max_train_time_steps=args.training_steps)
    os.mkdir(os.path.join(out_dir, 'final'))
    agent.save_model(os.path.join(out_dir, 'final'))

    