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

from agent.dqn_agent import DQNAgent
from agent.bootstrapped_agent import BootstrappedAgent
from agent.noisy_agent import NoisyAgent
from agent.ez_greedy_agent import EZGAgent
# from qlearn.toys.memory import ReplayBuffer
import matplotlib.pyplot as plt
from envs.nchain import NChainEnv_manyhot
import os

parser = argparse.ArgumentParser(description='DQN')
parser.add_argument('--seed', type=int, default=510, help='Random seed')
parser.add_argument('--cuda', type=int, default=0, help='use cuda')
parser.add_argument('--max-steps', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps')

parser.add_argument('--evaluation-episodes', type=int, default=1, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--replay_buffer_size', type=int, default=int(5e4), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--learning-freq', type=int, default=10, metavar='k', help='Frequency of sampling from memory')
parser.add_argument("--learning-starts", type=int, default=0, help="number of iterations after which learning starts")
parser.add_argument('--discount', type=float, default=0.999, metavar='GAMMA', help='Discount factor')
parser.add_argument('--target-update-freq', type=int, default=500, metavar='TAU', help='Number of steps after which to update target network')
parser.add_argument('--lr', type=float, default=0.0005, metavar='ETA', help='Learning rate')
parser.add_argument('--cls_lr', type=float, default=0.0001, metavar='ETA', help='Classifier Learning rate')
parser.add_argument('--adam-eps', type=float, default=1e-08, metavar='EPSILON', help='Adam epsilon')
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
parser.add_argument('--nchain-flip', type=int, default=0, help='whether or not ncahin environment flipped')
parser.add_argument('--u', type=float, default=2., help='u for zeta dist in ez-greedy')
parser.add_argument('--warm-up', type=int, default=256, help='model warm-up')




def run_experiment(args, input_dim, seed, u=None, final_exploration_step=1000):
    if final_exploration_step:
        args.final_exploration_step = final_exploration_step

    # Setup
    args = parser.parse_args()
    assert args.agent in ['DQN', 'BootstrappedDQN', 'NoisyDQN', 'ez_greedy']

    if input_dim:
        args.input_dim = input_dim
    if seed:
        args.seed = seed
    if u:
        args.u = u


    infos = dict()
    infos['chain_length'] = input_dim
    infos['total_rewards'] = []
    infos['episode_rewards'] = []
    infos['episode_decisions'] = []
    infos['timesteps'] = []
    infos['visits'] = []

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Environment

    env = NChainEnv_manyhot(args.input_dim)
    action_space = env.action_space.n


    # Agent
    if args.agent == 'BootstrappedDQN':
        dqn = BootstrappedAgent(args, env)
    elif args.agent == 'NoisyDQN':
        dqn = NoisyAgent(args, env)
    elif args.agent == 'DQN':
        dqn = DQNAgent(args, env)
    elif args.agent == 'ez_greedy':
        dqn = EZGAgent(args, env)
        n = 0
        action = 0

    replay_buffer = ReplayBuffer(args.replay_buffer_size)
    # mem = ReplayBuffer(args.memory_capacity)

    # schedule of epsilon annealing
    exploration = LinearSchedule(args.final_exploration_step, args.final_exploration, 1)

    # Training loop
    dqn.online_net.train()
    timestamp = 0
                        

    # Main Learning

    for episode in range(args.max_episodes):
        temp_visits = [0 for _ in range(args.input_dim)]
        temp_visits[1] = 1 # starting location
        # for episode info
        episode_reward = 0
        ed = 0


        if args.agent in ['BootstrappedDQN']:
            k = random.randrange(args.nheads)
        if args.agent in ["ez_greedy"]:
            n = 0

        epsilon = exploration.value(episode)

        state, done = env.reset(), False
        

        while not done:
            timestamp += 1
            if args.agent == 'BootstrappedDQN':
                ed += 1
                action = dqn.act_single_head(state[None], k)
            elif args.agent == 'NoisyDQN':
                ed += 1
                action = dqn.act(state[None], eval=False)
            elif args.agent == 'DQN':
                ed += 1
                action = dqn.act_e_greedy(state[None], epsilon=epsilon)
            elif args.agent == 'ez_greedy':
                if n == 0:
                    ed += 1
                n , action = dqn.act_ez_greedy(state[None], epsilon=epsilon, n=n , w=action)

            next_state, reward, done, _ = env.step(int(action))
            visited_state = int(sum(next_state)-1)
            temp_visits[visited_state] += 1
            episode_reward += reward
            # Store the transition in memory
            replay_buffer.add(state, action, reward, next_state, float(done))
    
            # Move to the next state
            state = next_state
            
            if timestamp % args.target_update_freq == 0:
                dqn.update_target_net()

        if timestamp > args.learning_starts:
            # train k(8) times
            for _ in range(1):
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size)
                '''if rewards[0]==1:
                    print("sample reward: {}".format(rewards[0]))'''
                if args.agent == 'BootstrappedDQN':
                    loss = dqn.learn(obses_t, actions, rewards, obses_tp1, dones)
                else:           
                    loss = dqn.learn(obses_t, actions, rewards, obses_tp1, dones) 
        
        print(f"episode {episode+1} reward: {round(episode_reward, 4)}")

        infos['total_rewards'].append(infos['total_rewards'][-1] + episode_reward) if infos['total_rewards'] else infos['total_rewards'].append(episode_reward) # append total rewards
        infos['episode_rewards'].append(episode_reward)
        infos['episode_decisions'].append(ed)
        infos['timesteps'].append(timestamp)
        infos['visits'].append(temp_visits)
        
        # if args.agent == "ez_greedy" and args.input_dim == 50:
        #     output_dir = f"{os.getcwd()}/qlearn/saved_models/{args.agent}/"
        #     final_path =f"{output_dir}/l{args.input_dim}_s{seed}_e2/"
        #     if not os.path.exists(output_dir):
        #         os.mkdir(os.path.dirname(output_dir))
        #     if not os.path.exists(final_path):
        #         os.mkdir(os.path.dirname(final_path))

        #     torch.save(dqn.online_net.state_dict(), f"{final_path}/online_net.pth")
        #     torch.save(dqn.target_net.state_dict(), f"{final_path}/target_net.pth")

    return infos

        
if __name__ == "__main__":
    args = parser.parse_args()
    assert args.agent in ['DQN', 'BootstrappedDQN', 'NoisyDQN', 'ez_greedy']
    rewards = run_experiment(args, args.input_dim, args.seed, u=args.u, final_exploration_step=args.input_dim*10)
