"""
Adaptation of the vanilla DDPG code to allow for UTE modification.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ucb import UCB

# we use exactly the same Actor and Critic networks and training methods for both as in the vanilla implementation
from DDPG.vanilla import DDPG as VanillaDDPG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Q(nn.Module):
    """
    Simple fully connected Q function. Also used for skip-Q when concatenating behaviour action and state together.
    Used for simpler environments such as mountain-car or lunar-lander.
    """

    def __init__(self, state_dim, action_dim, skip_dim):
        super(Q, self).__init__()
        # We follow the architecture of the Actor and Critic networks in terms of depth and hidden units
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, skip_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Extension_CoreNet(nn.Module):
    def __init__(self, state_dim, action_dim, ):
        super(Extension_CoreNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)

    def forward(self, x, action_val=None):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class Ensemble_Extension(nn.Module):
    def __init__(self, core_net, skip_dim=8, nheads=10):
        super(Ensemble_Extension, self).__init__()
        self.nheads = nheads
        self.core_net = core_net
        self.heads = nn.ModuleList([nn.Linear(300, skip_dim) for _ in range(self.nheads)])

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


class DDPG(VanillaDDPG):
    def __init__(self, state_dim, action_dim, max_action, skip_dim, discount=0.99, tau=0.005):
        # We can fully reuse the vanilla DDPG and simply stack TempoRL on top
        super(DDPG, self).__init__(state_dim, action_dim, max_action, discount, tau)

        # Create Skip Q network
        self.core_net = Extension_CoreNet(state_dim, action_dim)
        self.core_net_target = Extension_CoreNet(state_dim, action_dim)

        self.n_heads = 10
        self.skip_Q = Ensemble_Extension(self.core_net, skip_dim, self.n_heads).to(device)
        self.skip_optimizer = torch.optim.Adam(self.skip_Q.parameters())

        self.bernoulli_probability = 0.5
        self.skip_loss_function = nn.SmoothL1Loss()
        # For bandit: Adaptively choose uncertainty factor lambda
        self.lambdas = [-2.5, -1.5, -1.0, -0.5, 0.0, 0.5]
        num_arms = len(self.lambdas)
        window_size = 500
        ucb_epsilon = 0.1
        ucb_beta = 0.5
        self.ucb = UCB(num_arms, window_size, ucb_epsilon, ucb_beta)

    def select_skip(self, state, action):
        """
        Select the skip action.
        Has to be called after select_action
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(device)
        current_outputs = self.skip_Q(torch.cat([state, action], 1).to(device))
        outputs = []
        for k in range(self.n_heads):
            outputs.append(current_outputs[k].detach().cpu().numpy())
        outputs = np.array(outputs)
        mean_Q = np.mean(outputs , axis=0) # 1x10
        std_Q = np.std(outputs, axis=0)
        Q_tilda = mean_Q + self.uncertainty_factor*std_Q
        u = np.argmax(Q_tilda.flatten())
        return u

    def select_skip_greedy(self, state, action):
        """
        Select the skip action.
        Has to be called after select_action
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(device)
        current_outputs = self.skip_Q(torch.cat([state, action], 1).to(device))
        outputs = []
        for k in range(self.n_heads):
            outputs.append(current_outputs[k].detach().cpu().numpy())
        outputs = np.array(outputs)
        mean_Q = np.mean(outputs , axis=0) # 1x10
        std_Q = np.std(outputs, axis=0)
        Q_tilda = mean_Q 
        u = np.argmax(Q_tilda.flatten())
        return u

    def get_batch_skip(self, state, action):
        current_outputs = self.skip_Q(torch.cat([state, action], 1).to(device))
        outputs = []
        for k in range(self.n_heads):
            outputs.append(current_outputs[k].detach().cpu().numpy())  # Bx10
        outputs = np.array(outputs) # HxBx10
        mean_Q = np.mean(outputs, axis=0) # Bx10
        std_Q = np.std(outputs, axis=0)
        Q_tilda = mean_Q + self.uncertainty_factor*std_Q
        u = np.argmax(Q_tilda, axis=-1) #B
        return u

    def train_skip(self, replay_buffer, batch_size=100):
        """
        Train the skip network
        """
        # Sample replay buffer
        state, action, skip, next_state, reward, not_done = replay_buffer.sample(batch_size*2)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * torch.pow(self.discount, skip + 1) * target_Q).detach()
    
        # Get current Q estimate
        current_outputs = self.skip_Q(torch.cat([state, action], 1))
        masks = torch.bernoulli(torch.zeros((batch_size*2, self.n_heads), device=device) + self.bernoulli_probability)
        cnt_losses = []
        for k in range(self.n_heads):
            total_used = torch.sum(masks[:,k])
            if total_used > 0.0:
                current_prediction = current_outputs[k][torch.arange(batch_size*2).long(), skip.reshape(skip.size(0)).long()].reshape(-1,1)
                l1loss = self.skip_loss_function(current_prediction, target_Q.detach())
                full_loss = masks[:,k]*l1loss
                loss = torch.sum(full_loss/total_used)
                cnt_losses.append(loss)

        # Optimize the critic
        self.skip_optimizer.zero_grad()
        skip_loss = sum(cnt_losses)/self.n_heads
        skip_loss.backward()
        for param in self.skip_Q.core_net.parameters():
            if param.grad is None:
                pass
                #print("##### Skip Q Parameter with grad = None:", param.name)
            else:
                param.grad.data *= 1.0/float(self.n_heads)
        for param in self.skip_Q.parameters():
            if param.grad is None:
                pass
            else:
                param.grad.data.clamp_(-10.0, 10.0)

        self.skip_optimizer.step()


    # Action Q update based on double DQN with nstep target
    def train(self, replay_buffer, batch_size=100):
        # Sample replay buffer
        state, action, skip, next_state, reward, not_done = replay_buffer.sample(batch_size*3)

        current_skip_prediction = self.get_batch_skip(state, action) 
        batch_use = np.where(skip.squeeze().detach().cpu().numpy() <= current_skip_prediction)[0]
        tmp_batch_size = len(batch_use)
        if tmp_batch_size > batch_size:
            batch_use = batch_use[:batch_size]
            tmp_batch_size = len(batch_use)

        state = state[batch_use]
        action = action[batch_use]
        skip = skip[batch_use]
        next_state = next_state[batch_use]
        reward = reward[batch_use]
        not_done = not_done[batch_use]

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        super().save(filename)

        torch.save(self.skip_Q.state_dict(), filename + "_skip")
        torch.save(self.skip_optimizer.state_dict(), filename + "_skip_optimizer")

    def load(self, filename):
        super().load(filename)

        self.skip_Q.load_state_dict(torch.load(filename + "_skip"))
        self.skip_optimizer.load_state_dict(torch.load(filename + "_skip_optimizer"))
