import torch


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        if len(obs.shape) > 3:
            n_obs = obs.shape[0]
            action = [self.action_space.sample() for _ in range(n_obs)]
        else:
            action = self.action_space.sample()
        return action


class ActorCritic(object):
    def __init__(self, action_space, device):
        self.action_space = action_space
        self.device = device

    def train(self, states, actions, rewards, discounts):
        pass

    def act(self, obs, is_latent=False):
        if is_latent:
            n_obs = obs.shape[0]
            action = [self.action_space.sample() for _ in range(n_obs)]
            action = torch.tensor(action, dtype=torch.long, device=self.device)
        else:
            action = self.action_space.sample()
        return action
