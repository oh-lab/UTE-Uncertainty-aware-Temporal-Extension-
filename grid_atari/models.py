import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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