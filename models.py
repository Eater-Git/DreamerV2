import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class WorldModel(object):
    def __init__(self, n_latent, n_p, device):
        self.n_latent = n_latent
        self.n_p = n_p
        self.device = device

    def train(self, dataset):
        pass

    def sample_state(self, n_sample):
        state = torch.zeros([n_sample, self.n_latent + self.n_p], device=self.device)
        return state

    def imagine(self, agent, init_state, horizon):
        n_trajectory = init_state.size()[0]
        states = torch.zeros([n_trajectory, horizon, self.n_latent + self.n_p], device=self.device)
        actions = torch.zeros([n_trajectory, horizon], device=self.device)
        rewards = torch.zeros([n_trajectory, horizon], device=self.device)
        discounts = torch.zeros([n_trajectory, horizon], device=self.device)
        return states, actions, rewards, discounts


class ActorNetwork(nn.Module):
    def __init__(
      self, layers, num_of_actions, units, act):
        print("init")
        super(ActionHead, self).__init__()
        self._layers = layers
        self._num_of_actions = num_of_actions
        self._act = act
        self.fc = []
        for i in range(layers -1):
            self.fc.append(nn.Linear(units,units))
        
        self.fc.append(nn.Linear(units, num_of_actions))


    #param: features(z)
    #return: action distribution
    def forward(self, features):
        for i in range(self._layers - 1):
            features = self._act(self.fc[i](features))
        
        out = F.softmax(self.fc[-1](fetures))
        return out
