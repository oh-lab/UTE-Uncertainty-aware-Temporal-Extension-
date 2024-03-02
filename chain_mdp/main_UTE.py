import argparse
import time
from datetime import datetime
import random
import numpy as np
import pandas as pd
import seaborn as sns
import math
from collections import Counter
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import ReplayBuffer

from envs.nchain import NChainEnv_manyhot
import matplotlib.pyplot as plt

# for tempoRL and UTE
from itertools import count
from agent.temporal_extension import TDQN, UTE, soft_update
from agent.temporal_extension import ReplayBuffer as RB
from agent.temporal_extension import NoneConcatSkipReplayBuffer
# from evaluation_wrapper import collect_plot_info

import os


parser = argparse.ArgumentParser(description='DQN')
parser.add_argument('--seed', type=int, default=510, help='Random seed')
parser.add_argument('--cuda', type=int, default=1, help='use cuda')
parser.add_argument('--max-steps', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps')

parser.add_argument('--evaluation-episodes', type=int, default=1, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--replay_buffer_size', type=int, default=5e4, metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--learning-freq', type=int, default=10, metavar='k', help='Frequency of sampling from memory')
parser.add_argument("--learning-starts", type=int, default=0, help="number of iterations after which learning starts")
parser.add_argument('--discount', type=float, default=0.999, metavar='GAMMA', help='Discount factor')
parser.add_argument('--target-update-freq', type=int, default=500, metavar='TAU', help='Number of steps after which to update target network')
parser.add_argument('--lr', type=float, default=0.0005, metavar='ETA', help='Learning rate')
parser.add_argument('--cls_lr', type=float, default=0.0005, metavar='ETA', help='Classifier Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-6, metavar='EPSILON', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=64, metavar='SIZE', help='Batch size')
parser.add_argument('--input-dim', type=int, default=8, help='the length of chain environment')
parser.add_argument('--evaluation-interval', type=int, default=10, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--nheads', type=int, default=10, help='number of heads in Bootstrapped DQN')
parser.add_argument('--agent', type=str, default='DQN', help='type of agent')
parser.add_argument('--final-exploration', type=float, default=0.001, help='last value of epsilon')
parser.add_argument('--final-exploration-step', type=float, default=100, help='horizon of epsilon schedule')
parser.add_argument('--max-episodes', type=int, default=int(1000), metavar='EPISODES', help='Number of training episodes')
parser.add_argument('--hidden_dim', type=int, default=int(16), help='number of hidden unit used in normalizing flows')
parser.add_argument('--n-hidden', type=int, default=int(0), help='number of hidden layer used in normalizing flows')
parser.add_argument('--n-flows-q', type=int, default=int(1), help='number of normalizing flows using for the approximate posterior q')
parser.add_argument('--n-flows-r', type=int, default=int(1), help='number of normalizing flows using for auxiliary posterior r')
parser.add_argument('--double-q', type=int, default=1, help='whether or not to use Double DQN')
parser.add_argument('--skip-net-max-skips', type=int, default=10, help='Maximum skip-size')
parser.add_argument('--uncertainty-factor', type=float, default=2.0, help='hyperparam for uncertainty sensitivity')



def run_experiment(args, input_dim, seed, skip_net_max_skips, uncertainty_factor=0.0, final_exploration_step=1000):
    if input_dim:
        args.input_dim = input_dim
    if seed:
        args.seed = seed
    if skip_net_max_skips:
        args.skip_net_max_skips = skip_net_max_skips
    if uncertainty_factor:
        args.uncertainty_factor = uncertainty_factor
    if final_exploration_step:
        args.final_exploration_step = final_exploration_step

    infos = dict()
    infos['chain_length'] = input_dim
    infos['total_rewards'] = []
    infos['episode_rewards'] = []
    infos['episode_decisions'] = []
    infos['timesteps'] = []
    infos['visits'] = []
    infos['action'] = []
    infos['j'] = []

    # Setup
    device = torch.device('cpu')
    print("using cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Environment
    env =  NChainEnv_manyhot(args.input_dim)

    # Agent
    if args.agent in ['tdqn']:
        agent = TDQN(env.observation_space.n, env.action_space.n, args.skip_net_max_skips, gamma=args.discount)
    elif args.agent == 'ute':
        agent = UTE(env.observation_space.n, env.action_space.n, args.skip_net_max_skips, gamma=0.99, uncertainty_factor=args.uncertainty_factor)

    replay_buffer = RB(args.replay_buffer_size)
    skip_replay_buffer = NoneConcatSkipReplayBuffer(args.replay_buffer_size)

    # mem = ReplayBuffer(args.memory_capacity)

    # schedule of epsilon annealing
    exploration = LinearSchedule(args.final_exploration_step, args.final_exploration, 1)

    # Training loop
    timestamp = 0

    episode_reward = None
    # Main Learning
    for episode in range(args.max_episodes):
        # print("%s/%s" % (episode + 1, args.max_episodes))
        start = time.time()

        if args.agent in ['ute']:
            k = random.randrange(args.nheads)

        # if episode_reward and episode_reward == 10.0:
        #     stop_update = True

        # for episode info
        episode_reward = 0
        ed = 0

        epsilon = exploration.value(episode)

        state, done = env.reset(), False
        
        while not done:
            temp_visits = [0 for _ in range(args.input_dim)]
            temp_visits[1] = 1 # starting location
            # temp_actions = []
            # temp_j = []

            if args.agent in ['tdqn', 'ute']:
                ### train tdqn agent ###
                for _ in count():
                    if args.agent in ['ute']:
                        action = agent.get_action(state[None], epsilon)
                        skip = agent.get_skip(state[None], np.array([action]))  # get skip with the selected action as context
                    else:
                        action = agent.get_action(state[None], epsilon)
                        skip = agent.get_skip(state, np.array([action]), epsilon)  # get skip with the selected action as context
                    
                    d = False
                    skip_states, skip_rewards = [], []
                    ed += 1
                    for curr_skip in range(skip + 1):
                        # temp_actions.append(action)
                        # temp_j.append(skip)
                        ns, r, d, _ = env.step(int(action))
                        # print(episode, sum(state), action, episode_reward, epsilon)
                        visited_state = int(sum(ns)-1)
                        temp_visits[visited_state] += 1
                        episode_reward += r
                        timestamp += 1
                        skip_states.append(state)
                        skip_rewards.append(r)

                        # Update the skip replay buffer with all observed skips.
                        skip_id = 0
                        for start_state in skip_states:
                            skip_reward = 0
                            for exp, r in enumerate(skip_rewards[skip_id:]):  # make sure to properly discount
                                skip_reward += np.power(agent._gamma, exp) * r
                            # also keep track of the behavior action
                            skip_replay_buffer.add_transition(start_state, curr_skip - skip_id, ns,
                                                            skip_reward, d, curr_skip - skip_id + 1, np.array([action]))  
                            skip_id += 1
                        replay_buffer.add_transition(state, action, ns, r, d)

                        if d:
                            done  = d
                            break                    
                        state = ns

                    if d:
                        done = d
                        break

        if args.agent in ['tdqn', 'ute']:
            if timestamp > args.learning_starts and args.agent == 'ute':
                batch_states, batch_actions, batch_next_states, batch_rewards,\
                    batch_terminal_flags, batch_lengths, batch_behaviours = \
                    skip_replay_buffer.random_next_batch(args.batch_size*2)

                target = batch_rewards + (1 - batch_terminal_flags) * torch.pow(agent._gamma, batch_lengths) * \
                            agent._q_target(batch_next_states)[torch.arange(args.batch_size*2).long(), torch.argmax(
                                agent._q(batch_next_states), dim=1)]
                skip_states = torch.hstack([batch_states, batch_behaviours])
                current_outputs = agent._skip_q(skip_states)

                masks = torch.bernoulli(torch.zeros((args.batch_size*2, agent.n_heads), device=device) + agent.bernoulli_probability)
                cnt_losses = []
                for k in range(agent.n_heads):
                    total_used = torch.sum(masks[:,k])
                    if total_used > 0.0:
                        current_prediction = current_outputs[k][torch.arange(args.batch_size*2).long(), batch_actions.long()]
                        l1loss = agent._skip_loss_function(current_prediction, target.detach())
                        full_loss = masks[:,k]*l1loss
                        loss = torch.sum(full_loss/total_used)
                        cnt_losses.append(loss)
                
                agent._skip_q_optimizer.zero_grad()
                skip_loss = sum(cnt_losses)/agent.n_heads
                skip_loss.backward()
                agent._skip_q_optimizer.step()

                batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                    replay_buffer.random_next_batch(args.batch_size)

                target = batch_rewards + (1 - batch_terminal_flags) * agent._gamma * \
                        agent._q_target(batch_next_states)[torch.arange(args.batch_size).long(), torch.argmax(
                            agent._q(batch_next_states), dim=1)]
                current_prediction = agent._q(batch_states)[torch.arange(args.batch_size).long(), batch_actions.long()]

                loss = agent._loss_function(current_prediction, target.detach())

                agent._q_optimizer.zero_grad()
                loss.backward()
                agent._q_optimizer.step()

                soft_update(agent._q_target, agent._q, 0.5)

            elif timestamp > args.learning_starts  and args.agent == 'tdqn':
                # update is performed after the episode ends
                # Skip Q update based on double DQN where target is behavior Q
                batch_states, batch_actions, batch_next_states, batch_rewards, \
                batch_terminal_flags, batch_lengths, batch_behaviours = \
                    skip_replay_buffer.random_next_batch(args.batch_size)
                batch_states, batch_actions, batch_next_states, batch_rewards, \
                batch_terminal_flags, batch_lengths, batch_behaviours = \
                    batch_states.to(device), batch_actions.to(device), batch_next_states.to(
                        device), batch_rewards.to(device), \
                    batch_terminal_flags.to(device), batch_lengths.to(device), batch_behaviours.to(device)

                target = batch_rewards + (1 - batch_terminal_flags) * torch.pow(agent._gamma, batch_lengths) * \
                    agent._q_target(batch_next_states)[torch.arange(args.batch_size).long(), torch.argmax(agent._q(batch_next_states), dim=1)]
                current_prediction = agent._skip_q(batch_states, batch_behaviours)[torch.arange(args.batch_size).long(), batch_actions.long()]

                loss = agent._skip_loss_function(current_prediction, target.detach())

                agent._skip_q_optimizer.zero_grad()
                loss.backward()
                agent._skip_q_optimizer.step()

                # Action Q update based on double DQN with standard target network
                replay_buffer.add_transition(state, action, ns, r, d)
                batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                    replay_buffer.random_next_batch(args.batch_size)

                batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                    batch_states.to(device), batch_actions.to(device), batch_next_states.to(
                        device), batch_rewards.to(device), batch_terminal_flags.to(device)

                target = batch_rewards + (1 - batch_terminal_flags) * agent._gamma * \
                        agent._q_target(batch_next_states)[torch.arange(args.batch_size).long(), torch.argmax(
                            agent._q(batch_next_states), dim=1)]
                current_prediction = agent._q(batch_states)[torch.arange(args.batch_size).long(), batch_actions.long()]

                loss = agent._loss_function(current_prediction, target.detach())

                agent._q_optimizer.zero_grad()
                loss.backward()
                agent._q_optimizer.step()

                soft_update(agent._q_target, agent._q, 0.5)

        infos['total_rewards'].append(infos['total_rewards'][-1] + episode_reward) if infos['total_rewards'] else infos['total_rewards'].append(episode_reward) # append total rewards
        infos['episode_rewards'].append(episode_reward)
        infos['episode_decisions'].append(ed)
        infos['timesteps'].append(timestamp)
        infos['visits'].append(temp_visits)

        print(f"episode {episode+1} reward: {round(episode_reward, 4)}")

    return infos


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.agent in ['tdqn', 'ute']
    rewards = run_experiment(args, args.input_dim, args.seed, args.skip_net_max_skips, uncertainty_factor=args.uncertainty_factor, final_exploration_step=args.input_dim*10)
