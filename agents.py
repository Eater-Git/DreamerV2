import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        action = self.action_space.sample()
        return action


class ActorCritic(object):
    def __init__(self, action_space, device):
        self.action_space = action_space
        self.device = device

    def train(self, states, actions, rewards, discounts):
        pass

    def act(self, obs):
        action = self.action_space.sample()
        return action

    #L(\psi) = E_p[\Sigma_{t=1}^{H-1}(-\rho\ln p_\psi(a^_t|z^_t)sg(V^\lambda_t - v_\xi(z^_t)) - (1-\rho)V^\lambda_t - \etaH[a_t|z^_t])]
    def calcActorLoss(self):
        pass

    #L(\xi) = E_p[\Sigma_{t=1}^{H-1}1/2(v_\xi(z^_t) - sg(V^\lambda_t)))^2]
    def calcCriticLoss(self):
        pass

    def accumulateValue(self):
        pass

class ActorNetwork(nn.Module):
    def __init__(
      self, layers, num_of_actions, units, act):
        print("init")
        super(ActorNetwork, self).__init__()
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
        
        out = F.softmax(self.fc[-1](features))
        return out

class CriticNetwork(nn.Module):
    def __init__(
      self, input_size, layers, units, act):
        print("init")
        super(CriticNetwork, self).__init__()
        self._layers = layers
        self._inputsize = input_size
        self._act = act
        self.fc = []

        self.fc.append(nn.Linear(input_size,units))
        for i in range(layers -2):
            self.fc.append(nn.Linear(units,units))
        
        self.fc.append(nn.Linear(units, 1))


    #param: features(z)
    #return: Value function
    def forward(self, features):
        for i in range(self._layers - 1):
            features = self._act(self.fc[i](features))
        
        out = F.softmax(self.fc[-1](features))
        return out
