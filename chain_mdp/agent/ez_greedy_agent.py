import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from model import DQN

# From Temporally Extended Epsilon-Greedy Exploration: https://arxiv.org/abs/2006.01782
class EZGAgent():
    def __init__(self, args, env):
        self.action_space = env.action_space.n
        self.batch_size = args.batch_size
        self.discount = args.discount
        self.double_q = args.double_q
        self.u = args.u  # for zeta dist
        self.input_dim = args.input_dim

        self.online_net = DQN(args, self.action_space)
        if args.model and os.path.isfile(args.model):
            self.online_net.load_state_dict(torch.load(args.model))
        self.online_net.train()

        self.target_net = DQN(args, self.action_space)
        self.update_target_net()
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.lr, eps=args.adam_eps)
        if args.cuda:
            self.online_net.cuda()
            self.target_net.cuda()
        self.FloatTensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if args.cuda else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if args.cuda else torch.ByteTensor



    # Acts based on single state (no batch)
    def act(self, state):
        self.online_net.eval()
        state = Variable(self.FloatTensor(state))
        #print("{}: {} act:{}".format(torch.sum(state), self.online_net(state).data, self.online_net(state).data.max(1)[1][0]))
        return self.online_net(state).data.max(1)[1][0]

    # Acts with an EZ-greedy policy
    def act_ez_greedy(self, state, epsilon = 0.01, n=0, w=0):
        if n == 0:
            if random.random() < epsilon:
                n = min(int(np.random.zipf(self.u, 1)), self.input_dim ) # zeta dist
                w = random.randrange(self.action_space)
                action = w
            else:    
                action = self.act(state)
        else:
            action = w
            n -= 1
        return n, action

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def learn(self, states, actions, rewards, next_states, terminals):
        self.online_net.train()
        self.target_net.eval()
        states = Variable(self.FloatTensor(states))
        actions = Variable(self.LongTensor(actions))
        next_states = Variable(self.FloatTensor(next_states))
        rewards = Variable(self.FloatTensor(rewards)).view(-1, 1)
        terminals = Variable(self.FloatTensor(terminals)).view(-1, 1)

        state_action_values = self.online_net(states).gather(1, actions.view(-1, 1))
        if self.double_q:
            next_actions = self.online_net(next_states).max(1)[1]
            next_state_values = self.target_net(next_states).gather(1, next_actions.view(-1, 1))
        else:
            next_state_values = self.target_net(next_states).max(1)[0]

        # Compute V(s_{t+1}) for all next states.
        target_state_action_values = rewards + (1 - terminals) * self.discount * next_state_values.view(-1, 1)
        # Undo volatility (which was used to prevent unnecessary gradients)
        #target_state_action_values = Variable(target_state_action_values.data)

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, target_state_action_values.detach())

        # Optimize the model
        self.optimiser.zero_grad()
        loss.backward()
        for param in self.online_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimiser.step()
        return loss