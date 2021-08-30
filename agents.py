import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        action = self.action_space.sample()
        return action


class ActorCritic(object):
    def __init__(self, layers,num_of_actions, units, act, device):
        self.num_of_actions = num_of_actions
        self.device = device

        self.actor = ActorNetwork(layers, num_of_actions, units, act)

    def train(self, states, actions, rewards, discounts):
        values = accumulateValue(states, rewards, discounts)

        critic_loss = calcCriticLoss(values, self.critic)
        critic_loss.backward()

        actor_loss = calcActorLoss(states, actions, values, self.critic)
        actor_loss.backward()

    def act(self, obs):
        m = Categorical(self.actor.forward(obs))
        action = m.sample()
        return action.item()

    #L(\psi) = E_p[\Sigma_{t=1}^{H-1}(-\rho\ln p_\psi(a^_t|z^_t)sg(V^\lambda_t - v_\xi(z^_t)) - (1-\rho)V^\lambda_t - \etaH[a_t|z^_t])]
    def calcActorLoss(self, feat, action, target, weight, mix = 0.5, ent_scale):
        with torch.no_grad():
            policy = self.actor(feat)
        if self.config.actor_grad == 'dynamics':
            objective = target
        elif self.config.actor_grad == 'reinforce':
            baseline = self.critic(feat[:-1]).mode()
            with torch.no_grad():
                advantage = target - baseline
            objective = policy.log_prob(action)[:-1] * advantage
        elif self.config.actor_grad == 'both':
            baseline = self.critic(feat[:-1]).mode()
            with torch.no_grad():
                advantage = target - baseline
            objective = policy.log_prob(action)[:-1] * advantage
            # mix = common.schedule(self.config.actor_grad_mix, self.step)
            objective = mix * target + (1 - mix) * objective
        else:
            raise NotImplementedError(self.config.actor_grad)
        ent = policy.entropy()
        # ent_scale = common.schedule(self.config.actor_ent, self.step)
        objective += ent_scale * ent[:-1]
        actor_loss = -(weight[:-1] * objective).mean()
        return actor_loss

    #L(\xi) = E_p[\Sigma_{t=1}^{H-1}1/2(v_\xi(z^_t) - sg(V^\lambda_t)))^2]
    def calcCriticLoss(self, feat, action, target, weight):
        # _target = None
        
        dist = self.critic(feat)[:-1]
        with torch.no_grad():
            _target = target
        critic_loss = -(dist.log_prob(_target) * weight[:-1]).mean()
        return critic_loss

    #v_\xi
    def accumulateValue(self, states, rewards, discounts):
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
