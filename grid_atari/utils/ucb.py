import collections
import random

import numpy as np
import torch

class UCB:
    """
    Determine the index of the arms in terms of solving a multi-armed bandit problem
    Attributes:
      data           : list that stores the index and average reward of the arms
      num_arms  (int): number of arms used in multi-armed bandit problem
      epsilon (float): probability to select the index of the arms used in multi-armed bandit problem
      beta    (float): weight between frequency and mean reward
      count     (int): if count is less than num_arms, index is count because of trying to pick every arm at least once
    """

    def __init__(self, num_arms, window_size, epsilon, beta):
        """
        num_arms    (int): number of arms used in multi-armed bandit problem
        window_size (int): size of window used in multi-armed bandit problem
        epsilon   (float): probability to select the index of the arms used in multi-armed bandit problem
        beta      (float): weight between frequency and mean reward
        """
        
        self.data = collections.deque(maxlen=window_size)
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.beta = beta
        self.count = 0

    def pull_index(self):
        """
        pull index to determine value of betas and gammas
        Returns:
          index (float): index of arms 
        """
        
        if self.count < self.num_arms:
            index = self.count
            self.count += 1
            
        else:
            if random.random() > self.epsilon:
                N = np.zeros(self.num_arms)
                mu = np.zeros(self.num_arms)
                
                for j, reward in self.data:
                    N[j] += 1
                    mu[j] += reward
                mu = mu / (N + 1e-10)
                index = np.argmax(mu + self.beta * np.sqrt(1 / (N + 1e-6)))
                
            else:
                index = np.random.choice(self.num_arms)
        return index

    def push_data(self, datas):
        """
        push datas to UCB's data list
        Args:
          datas :store index of arms and resulting reward         
        """
        
        self.data += [(j, reward) for j, reward in datas]