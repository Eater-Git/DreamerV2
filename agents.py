import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import re


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    assert np.ndim(reward.shape) == np.ndim(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones(reward) #tf.ones_like(reward)
    dims = list(range(reward.shape.ndims))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
    if axis != 0:
        reward = torch.transpose(reward, dims)
        value = torch.transpose(value, dims)
        pcont = torch.transpose(pcont, dims)
    if bootstrap is None:
        bootstrap = torch.zeros(value[-1])
    next_values = torch.concat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    returns = static_scan(
        lambda agg, cur: cur[0] + cur[1] * lambda_ * agg,
        (inputs, pcont), bootstrap, reverse=True)
    if axis != 0:
        returns = torch.transpose(returns, dims)
    return returns

def static_scan(fn, inputs, start, reverse=False):
    last = start
    outputs = [[] for _ in torch.flatten(start)]
    indices = range(torch.flatten(inputs)[0].shape[0])
    if reverse:
        indices = reversed(indices)
    for index in indices:
        inp = torch.tensor(map(lambda x: x[index], inputs))
        last = fn(last, inp)
        [o.append(l) for o, l in zip(outputs, torch.flatten(last))]
    if reverse:
        outputs = [list(reversed(x)) for x in outputs]
    outputs = [torch.stack(x, 0) for x in outputs]
    return torch.reshape(outputs, start.shape) # return tf.nest.pack_sequence_as(start, outputs)


def schedule(string, step):
    try:
        return float(string)
    except ValueError:
        # step = tf.cast(step, tf.float32)
        match = re.match(r'linear\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clamp(step / duration, 0, 1)
            return (1 - mix) * initial + mix * final
        match = re.match(r'warmup\((.+),(.+)\)', string)
        if match:
            warmup, value = [float(group) for group in match.groups()]
            scale = torch.clamp(step / warmup, 0, 1)
            return scale * value
        match = re.match(r'exp\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, halflife = [float(group) for group in match.groups()]
            return (initial - final) * 0.5 ** (step / halflife) + final
        match = re.match(r'horizon\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clamp(step / duration, 0, 1)
            horizon = (1 - mix) * initial + mix * final
            return 1 - 1 / horizon
        raise NotImplementedError(string)

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        action = self.action_space.sample()
        return action


class ActorCritic(object):
    def __init__(self, act_layers,num_of_actions, act_units, input_size, critic_layers, critic_units, act, device):
        self.horizon = 100
        self.num_of_actions = num_of_actions
        self.device = device

        self.actor = ActorNetwork(act_layers, num_of_actions, act_units, act)
        self.critic = CriticNetwork(input_size, critic_layers,critic_units, act)
        self._target_critic = CriticNetwork(input_size, critic_layers,critic_units, act)

    def train(self, env, start, optimizerA, optimizerC):
        # feat, state, action, disc = imagine(self.actor, start, horizon)
        actions = []
        observations = []
        rewards = []
        dones = []

        observation = env.reset()
        observations.append(observation)
        for i in range(self.horizon):
            act = self.act(torch.tensor(observation, dtype=torch.float))
            observation, reward, done, info = env.step(act)
            actions.append(act)
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
        actions = torch.Tensor(actions)
        observations = torch.Tensor(observations)
        rewards = torch.Tensor(rewards)
        
        def done2disc(dones):
            disc = [0.999 if done is False else 0 for done in dones]
            return disc
        
        disc = torch.Tensor(done2disc(dones))
        
        target, weight = self.target(observations, actions, rewards, disc)

        optimizerC.zero_grad()
        optimizerA.zero_grad()
        critic_loss = calcCriticLoss(observations, actions, target, weight)
        critic_loss.backward()

        actor_loss = calcActorLoss(observations, actions, target, weight)
        actor_loss.backward()
        optimizerC.step()
        optimizerA.step()

    def act(self, obs):
        m = Categorical(self.actor.forward(obs))
        action = m.sample()
        return action.item()

    #L(\psi) = E_p[\Sigma_{t=1}^{H-1}(-\rho\ln p_\psi(a^_t|z^_t)sg(V^\lambda_t - v_\xi(z^_t)) - (1-\rho)V^\lambda_t - \etaH[a_t|z^_t])]
    def calcActorLoss(self, feat, action, target, weight, mix = 0.5):
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
            mix = schedule(self.config.actor_grad_mix, self.step)
            objective = mix * target + (1 - mix) * objective
        else:
            raise NotImplementedError(self.config.actor_grad)
        ent = policy.entropy()
        ent_scale = schedule(self.config.actor_ent, self.step)
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

    def target(self, feat, action, reward, disc):
        value = self._target_critic(feat).mode()
        target = lambda_return(
            reward[:-1], value[:-1], disc[:-1],
            bootstrap=value[-1], lambda_=0.9, axis=0)
        with torch.no_grad():
            weight = torch.cumprod(torch.cat(
                [torch.ones(disc[:1]), disc[:-1]], 0), 0)

        return target, weight

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
        self.fc = nn.ModuleList(self.fc)


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
        self.fc = nn.ModuleList(self.fc)


    #param: features(z)
    #return: Value function
    def forward(self, features):
        if isinstance(features, torch.Tensor) is False:
            features = torch.Tensor(features)
        for i in range(self._layers - 1):
            features = self._act(self.fc[i](features))
        
        out = F.softmax(self.fc[-1](features))
        return torch.Tensor(out)
